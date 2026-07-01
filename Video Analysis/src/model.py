import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops import roi_align

# Assuming config.py exists and defines BACKBONE_OUT_FEATURES
from config import *

class MotionVectorEncoder(nn.Module):
    """A small network to process raw motion vectors into a fixed-size embedding."""
    def __init__(self, input_dim=2, embedding_dim=64):
        super(MotionVectorEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
    
    def forward(self, mv_sequence):
        batch_size, seq_len, num_vectors, _ = mv_sequence.shape
        mv_sequence = mv_sequence.view(-1, 2)
        embeddings = self.mlp(mv_sequence)
        embeddings = embeddings.view(batch_size * seq_len, num_vectors, -1)
        pooled_embeddings, _ = torch.max(embeddings, dim=1)
        final_sequence = pooled_embeddings.view(batch_size, seq_len, -1)
        return final_sequence

class LRATF(nn.Module):
    def __init__(self, num_classes=1):
        super(LRATF, self).__init__()

        # --- Visual Feature Extraction ---
        self.feature_extractor = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
        
        # This value should be updated based on the output channels of your chosen backbone
        backbone_out_features = BACKBONE_OUT_FEATURES
        
        # --- Regional LSTMs (10 regions) ---
        self.upper_lip_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.lower_lip_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.left_eye_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.right_eye_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.nose_bridge_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.left_cheek_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.right_cheek_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.left_eyebrow_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.right_eyebrow_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.chin_jawline_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        
        # --- Motion Vector Stream ---
        self.mv_encoder = MotionVectorEncoder(input_dim=2, embedding_dim=64)
        self.mv_lstm = nn.LSTM(input_size=64, hidden_size=32, bidirectional=True, batch_first=True)

        # --- Multimodal Fusion with Attention ---
        self.attention_embedding_dim = 128
        # Visual streams have an output of 128 (64 * 2) from bidirectional LSTM
        self.visual_proj = nn.Linear(128, self.attention_embedding_dim) 
        # Motion stream has an output of 64 (32 * 2) from bidirectional LSTM
        self.motion_proj = nn.Linear(64, self.attention_embedding_dim)

        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.attention_embedding_dim,
            num_heads=4,
            # dropout=0.1, 
            batch_first=True 
        )

        # --- Final Classifier ---
        # 10 visual streams + 1 motion stream = 11 streams
        # Each stream has a dimension of attention_embedding_dim (128)
        self.classifier = nn.Sequential(
            nn.Linear(11 * self.attention_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def get_region_boxes(self, landmarks, img_shape):
        """
        Calculates bounding boxes for 10 facial regions based on landmarks.
        The landmark indices are based on MediaPipe's 468-point facial landmark model.
        """
        h, w = img_shape
        
        # Define landmark indices for each of the 10 regions
        # NOTE: These indices are a suggestion and may need fine-tuning.
        regions = [
            landmarks[:, [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], :2],  # Upper Lip
            landmarks[:, [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291], :2],  # Lower Lip
            landmarks[:, [362, 382, 381, 380, 374, 373, 390, 249, 263], :2],  # Left Eye
            landmarks[:, [33, 7, 163, 144, 145, 153, 154, 155, 133], :2],  # Right Eye
            landmarks[:, [6, 197, 195, 5, 4, 1, 19, 94, 2], :2],  # Nose Bridge & Tip
            landmarks[:, [200, 426, 431, 411, 444, 396, 266, 350, 446], :2],  # Left Cheek
            landmarks[:, [45, 234, 248, 243, 44, 107, 59, 10, 336], :2],  # Right Cheek
            landmarks[:, [276, 283, 282, 295, 300, 293], :2],  # Left Eyebrow
            landmarks[:, [46, 53, 52, 65, 70, 63], :2],  # Right Eyebrow
            landmarks[:, [152, 172, 176, 148, 150, 149, 176, 172, 152], :2]  # Chin & Jawline
        ]
        
        boxes = []
        for region_pts in regions:
            x_coords, y_coords = region_pts[:, :, 0] * w, region_pts[:, :, 1] * h
            boxes.append(torch.stack([torch.min(x_coords, dim=1)[0], 
                                     torch.min(y_coords, dim=1)[0], 
                                     torch.max(x_coords, dim=1)[0], 
                                     torch.max(y_coords, dim=1)[0]], dim=1))
        return boxes

    def forward(self, x, landmarks, motion_vectors):
        batch_size, seq_len, _, h, w = x.shape
        regional_features_seq = [[] for _ in range(10)]

        for t in range(seq_len):
            frame = x[:, t] 
            frame_landmarks = landmarks[:, t]
            features = self.feature_extractor(frame)
            
            boxes = self.get_region_boxes(frame_landmarks, (h, w))
            box_indices = torch.arange(batch_size, device=x.device).view(-1, 1)
            
            for i in range(10): # Iterate over 10 regions
                rois = torch.cat([box_indices, boxes[i]], dim=1).float()
                aligned_features = roi_align(features, rois, output_size=(1, 1), spatial_scale=1.0/16.0).squeeze(-1).squeeze(-1)
                regional_features_seq[i].append(aligned_features)
        
        sequences = [torch.stack(seq, dim=1) for seq in regional_features_seq]
        
        # Pass each regional sequence through its corresponding LSTM
        _, (h_upper_lip, _) = self.upper_lip_lstm(sequences[0])
        _, (h_lower_lip, _) = self.lower_lip_lstm(sequences[1])
        _, (h_left_eye, _) = self.left_eye_lstm(sequences[2])
        _, (h_right_eye, _) = self.right_eye_lstm(sequences[3])
        _, (h_nose_bridge, _) = self.nose_bridge_lstm(sequences[4])
        _, (h_left_cheek, _) = self.left_cheek_lstm(sequences[5])
        _, (h_right_cheek, _) = self.right_cheek_lstm(sequences[6])
        _, (h_left_eyebrow, _) = self.left_eyebrow_lstm(sequences[7])
        _, (h_right_eyebrow, _) = self.right_eyebrow_lstm(sequences[8])
        _, (h_chin_jawline, _) = self.chin_jawline_lstm(sequences[9])
        
        # Process motion vectors
        mv_embeddings = self.mv_encoder(motion_vectors)
        _, (h_mv, _) = self.mv_lstm(mv_embeddings)

        # Concatenate hidden states from both directions for each stream
        h_streams = [
            torch.cat((h_upper_lip[-2,:,:], h_upper_lip[-1,:,:]), dim=1),
            torch.cat((h_lower_lip[-2,:,:], h_lower_lip[-1,:,:]), dim=1),
            torch.cat((h_left_eye[-2,:,:], h_left_eye[-1,:,:]), dim=1),
            torch.cat((h_right_eye[-2,:,:], h_right_eye[-1,:,:]), dim=1),
            torch.cat((h_nose_bridge[-2,:,:], h_nose_bridge[-1,:,:]), dim=1),
            torch.cat((h_left_cheek[-2,:,:], h_left_cheek[-1,:,:]), dim=1),
            torch.cat((h_right_cheek[-2,:,:], h_right_cheek[-1,:,:]), dim=1),
            torch.cat((h_left_eyebrow[-2,:,:], h_left_eyebrow[-1,:,:]), dim=1),
            torch.cat((h_right_eyebrow[-2,:,:], h_right_eyebrow[-1,:,:]), dim=1),
            torch.cat((h_chin_jawline[-2,:,:], h_chin_jawline[-1,:,:]), dim=1),
            torch.cat((h_mv[-2,:,:], h_mv[-1,:,:]), dim=1)
        ]

        # Project streams to a common embedding space for attention
        proj_streams = [self.visual_proj(h_streams[i]) for i in range(10)]
        proj_streams.append(self.motion_proj(h_streams[10]))
        
        # Stack streams to form the attention input
        attention_input = torch.stack(proj_streams, dim=1)
        
        # Apply Multi-head Self-Attention
        attention_output, _ = self.attention_layer(attention_input, attention_input, attention_input)
        
        # Flatten the output for the final classifier
        final_representation = attention_output.flatten(start_dim=1)
        
        # Final classification
        output = self.classifier(final_representation)
        return output

if __name__ == '__main__':
    # Example usage:
    # This assumes BACKBONE_OUT_FEATURES is defined, e.g., 576 for MobileNetV3 small
    BACKBONE_OUT_FEATURES = 576 
    
    # Create a dummy config module for testing
    import types
    config = types.ModuleType('config')
    config.BACKBONE_OUT_FEATURES = BACKBONE_OUT_FEATURES
    
    # Replace the import with the dummy config
    import sys
    sys.modules['config'] = config
    
    # Create a dummy input
    batch_size = 2
    seq_len = 5
    img_h, img_w = 224, 224
    
    # Dummy video input (batch_size, seq_len, channels, height, width)
    dummy_video = torch.randn(batch_size, seq_len, 3, img_h, img_w)
    
    # Dummy landmarks (batch_size, seq_len, num_landmarks, 2) - normalized coords
    dummy_landmarks = torch.rand(batch_size, seq_len, 468, 2)
    
    # Dummy motion vectors (batch_size, seq_len, num_mvs, 2)
    dummy_mvs = torch.randn(batch_size, seq_len, 100, 2)
    
    # Instantiate the model
    model = LRATF(num_classes=1)
    
    # Forward pass
    output = model(dummy_video, dummy_landmarks, dummy_mvs)
    
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, 1)
    print("Model forward pass successful!")