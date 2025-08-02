import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.ops import roi_align
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

    
        self.feature_extractor = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
        
        backbone_out_features = BACKBONE_OUT_FEATURES
        
        
        self.mouth_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.nose_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.eyes_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.eyebrows_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.chin_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        
        self.mv_encoder = MotionVectorEncoder(input_dim=2, embedding_dim=64)
        self.mv_lstm = nn.LSTM(input_size=64, hidden_size=32, bidirectional=True, batch_first=True)

        self.attention_embedding_dim = 128
        self.visual_proj = nn.Linear(128, self.attention_embedding_dim) 
        self.motion_proj = nn.Linear(64, self.attention_embedding_dim)

        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.attention_embedding_dim,
            num_heads=4,
            # dropout=0.1, # Moderate dropout
            batch_first=True 
        )

       
        self.classifier = nn.Sequential(
            nn.Linear(6 * self.attention_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def get_region_boxes(self, landmarks, img_shape):
        h, w = img_shape
        left_eyebrow_pts = landmarks[:, 63:70, :2]
        right_eyebrow_pts = landmarks[:, 293:300, :2]
        regions = [
            landmarks[:, 61:80, :2],
            landmarks[:, 291:300, :2],
            landmarks[:, 362:388, :2],
            torch.cat([left_eyebrow_pts, right_eyebrow_pts], dim=1),
            landmarks[:, [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 397], :2]
        ]
        boxes = []
        for region_pts in regions:
            x_coords, y_coords = region_pts[:, :, 0] * w, region_pts[:, :, 1] * h
            boxes.append(torch.stack([torch.min(x_coords, dim=1)[0], torch.min(y_coords, dim=1)[0], torch.max(x_coords, dim=1)[0], torch.max(y_coords, dim=1)[0]], dim=1))
        return boxes

    def forward(self, x, landmarks, motion_vectors):
        batch_size, seq_len, _, h, w = x.shape
        regional_features_seq = [[] for _ in range(5)]

        for t in range(seq_len):
            frame = x[:, t] 
            frame_landmarks = landmarks[:, t]
            features = self.feature_extractor(frame)
            
            boxes = self.get_region_boxes(frame_landmarks, (h, w))
            box_indices = torch.arange(batch_size, device=x.device).view(-1, 1)
            
            for i in range(5):
                rois = torch.cat([box_indices, boxes[i]], dim=1).float()
                aligned_features = roi_align(features, rois, output_size=(1, 1), spatial_scale=1.0/16.0).squeeze(-1).squeeze(-1)
                regional_features_seq[i].append(aligned_features)
        
        sequences = [torch.stack(seq, dim=1) for seq in regional_features_seq]
        
        _, (h_mouth, _) = self.mouth_lstm(sequences[0])
        _, (h_nose, _) = self.nose_lstm(sequences[1])
        _, (h_eyes, _) = self.eyes_lstm(sequences[2])
        _, (h_eyebrows, _) = self.eyebrows_lstm(sequences[3])
        _, (h_chin, _) = self.chin_lstm(sequences[4])
        
        mv_embeddings = self.mv_encoder(motion_vectors)
        _, (h_mv, _) = self.mv_lstm(mv_embeddings)

        h_streams = [
            torch.cat((h_mouth[-2,:,:], h_mouth[-1,:,:]), dim=1),
            torch.cat((h_nose[-2,:,:], h_nose[-1,:,:]), dim=1),
            torch.cat((h_eyes[-2,:,:], h_eyes[-1,:,:]), dim=1),
            torch.cat((h_eyebrows[-2,:,:], h_eyebrows[-1,:,:]), dim=1),
            torch.cat((h_chin[-2,:,:], h_chin[-1,:,:]), dim=1),
            torch.cat((h_mv[-2,:,:], h_mv[-1,:,:]), dim=1)
        ]

        proj_streams = [
            self.visual_proj(h_streams[0]),
            self.visual_proj(h_streams[1]),
            self.visual_proj(h_streams[2]),
            self.visual_proj(h_streams[3]),
            self.visual_proj(h_streams[4]),
            self.motion_proj(h_streams[5])
        ]
        
        attention_input = torch.stack(proj_streams, dim=1)
        attention_output, _ = self.attention_layer(attention_input, attention_input, attention_input)
        
        final_representation = attention_output.flatten(start_dim=1)
        
        output = self.classifier(final_representation)
        return output
