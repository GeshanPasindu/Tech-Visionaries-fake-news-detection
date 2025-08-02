import os
# import csv
import json
import yt_dlp

# 50 deepfake video URLs
deepfake_urls = [
    "https://youtu.be/oxXpB9pSETo?si=1xQqXxkUqaFMkhKD",
    "https://youtu.be/iyiOVUbsPcM?si=YNzOkuQ7JWy7TMhe",
    "https://youtu.be/1OqFY_2JE1c?si=sytMQfe8QXYlG5t7",
    "https://youtu.be/1OqFY_2JE1c?si=E5ddTJBgvU0HecQg",
    "https://youtube.com/shorts/ErXgb4cAVIM?si=9ETMIzq-UhwlY5NM",
    "https://youtube.com/shorts/LWP5wv4CTSQ?si=1gYm7ygQ4sZMHBEU",
    "https://youtube.com/shorts/IUBUe7ue2BY?si=XpgZ2zOmyN0Ci9TJ",
    "https://youtu.be/PINeQV0LH6k?si=Kn_Snn-BJHRk_lSz",
    "https://www.youtube.com/watch?v=kjI-JaRWG7s",
    "https://youtu.be/S1MBVXkQbWU?si=qNSpTWh_Say7NTS5",
    "https://www.youtube.com/watch?v=S1MBVXkQbWU",
    "https://youtube.com/shorts/FPPlB2FrIww?si=mjAnxLCmwFBijgRz",
    "https://youtube.com/shorts/nejvGU5Sjws?si=OaGe7Kx5k3r71ZG-",
    "https://youtube.com/shorts/QFBBj7gLZMA?si=pjW-geecnwKLTEvT",
    "https://youtu.be/IvY-Abd2FfM?si=7N4hlPih3guOmFUS",
    "https://www.youtube.com/watch?v=fqye4Cw0EF0",
    "https://youtube.com/shorts/zXqjFuyq3xk?si=vTzVQLFHtXTd9DDJ",
    "https://www.youtube.com/watch?v=1h-yy3h1u04",
    "https://youtube.com/shorts/oPbuyJqSQ2k?si=WGlqq_3FFAxPjejQ",
    "https://youtube.com/shorts/AdfuRTQ_zRY?si=6qNNYvCGNXnMZl8T",
    "https://youtube.com/shorts/M0T9hSQ2BF4?si=EOiFmrm7PWk6GVMU",
    "https://youtube.com/shorts/QYBa3tdfn_M?si=WnxGYYeGSV76_OD1",
    "https://youtube.com/shorts/b5OOx5dW8Ms?si=c6AwDO59PK5BHJg3",
    "https://youtube.com/shorts/LG3PSb1-iqE?si=EywABL9rT90Kjatg",
    "https://www.youtube.com/watch?v=sX8yteGXYes",
    "https://youtube.com/shorts/i6aOHtMgJlM?si=nOy2FQvxKMbZKmcz",
    "https://youtube.com/shorts/e2XNqJGX634?si=A9RMcRyFYCRzqKsQ",
    "https://youtube.com/shorts/d_kywv7NwOY?si=69g-VMe2aRsvGS62",
    "https://youtube.com/shorts/2SlDLv-CfHY?si=cmJVEexQ4aFlD4DC",
    "https://youtube.com/shorts/TnuDI4GwWLA?si=dfL1sHq0ZzB6xxzh",
    "https://youtube.com/shorts/MFSxylb1aCU?si=_KZdUIg78CeROCFJ",
    "https://youtube.com/shorts/bf6xpLM_p0o?si=7HqpdlkkqH1tuzKW",
    "https://youtube.com/shorts/bf6xpLM_p0o?si=aJqvSog-g6UAAz5a",
    "https://youtube.com/shorts/SwIkPAGHEGs?si=mkbgyggjnLlxPwa8",
    "https://youtube.com/shorts/1_bZWso5G0w?si=uR4S5MxYjgD7a0We",
    "https://youtube.com/shorts/LYsNRtEZRV4?si=Viu5dE-ZqDemYvXs",
    "https://youtube.com/shorts/AsclBdTj_EA?si=vmFhgVZb3l6gd3vS",
    "https://youtube.com/shorts/NzQPtQoI6uI?si=vsYTf9thrE25QIwA",
    "https://youtube.com/shorts/qjAvf3ccxUQ?si=6h8epJnvpKIYbYOP",
    "https://youtube.com/shorts/zZ1qiK9_mhE?si=9SxAG5b6V3GbHGl6",
    "https://youtu.be/A8TmqvTVQFQ?si=JzfgfdsxGve5XLOK",
    "https://youtu.be/XuKUkyPegBE?si=URzyH4F3gqOpC0vh",
    "https://youtu.be/SKlthQfN6io?si=pMXixLFE8qLLNXE0",
    "https://youtu.be/3BGRGeH-BP8?si=0HDP-ZiuT4HfwlgX",
    "https://youtu.be/PSL4KdTUFKU?si=5BQO6xExzHyEf28e",
    "https://youtu.be/LTQfBfvpv_k?si=EBWl_iWWGBvUWkKe",
    "https://youtu.be/pnPgfscR5gA?si=bIR1Uuwc2VOLxlIn",
    "https://www.youtube.com/shorts/j0WjIm2Fbuk",
    "https://youtu.be/R43dwcplFO4?si=sqmq4COfcrUYp-uz",
    "https://youtu.be/8OJnkJqkyio?si=wMdytw7bd7lr11KN",
    "https://youtu.be/3ntguiZab3s?si=R2sv7pDKYaug41Ax",
    "https://youtu.be/LTQfBfvpv_k?si=VyBqfH48KmOedExr",
    "https://youtu.be/HN-qlGf2mZw?si=sr60r1YRGFRxSEEh",
    "https://youtu.be/oxVcokRghpE?si=gRCTNGXp1jBHswAi"
]

# 50 real interview video URLs
real_urls = [
    "https://www.youtube.com/watch?v=WLy8HteAUlM",
    "https://www.youtube.com/watch?v=TTuf6dseaXU",
    "https://youtu.be/OHCEjPKsk40?si=uaeBzOkdMv7R7ePi",
    "https://youtu.be/wIyj1XupNXs?si=Dz0JmnxX2fL9hmk1",
    "https://youtu.be/Ocm0GpUIG0Q?si=nrI3KQwRR_ijkDfK",
    "https://youtu.be/ocGJWc2F1Yk?si=E5ct_NSX4xywpXqU",
    "https://youtu.be/EaJtMAWwWVg?si=0JJCZtJYBdX9kXmO",
    "https://youtu.be/Ny-qhl4N9dY?si=oCmi-Evyx2YerkyF",
    "https://youtu.be/vWYONbP7DL0?si=emxx6hTce_sJgy8l",
    "https://youtu.be/dmbpagijVkk?si=2GDdisd-b5cGBWx4",
    "https://youtube.com/shorts/TnVc3KWkl28?si=FSr_BcjtkQ1_gAGq",
    "https://youtube.com/shorts/3cbca8llmQI?si=V3OE_WjBCZ4Jcxb0",
    "https://youtube.com/shorts/v9Cn3GQicS4?si=3ptDlsVM_ZmAvcFD",
    "https://youtu.be/pCQ9r7lguug?si=vamZEG7_4YSCf5Q2",
    "https://youtu.be/3JpaT5M4TAI?si=11axOoR7e_JHoWWs",
    "https://youtu.be/2JiXLxoveQg?si=aIwWNOck5o1T7BaX",
    "https://youtu.be/ES-U1huVBEY?si=Hqg3zJXVV_nv5TyI",
    "https://youtu.be/2eB994E7z2c?si=RML2DXGCiFJI--dB",
    "https://youtu.be/pXBaWI7KRyk?si=paL_sfTjAMo3M0vf",
    "https://youtube.com/shorts/7cXW6Neb61c?si=9Mmss38Bb-_JQjiY",
    "https://youtu.be/0r-QzUH7ia0?si=alLZ5NufM96z6HVQ",
    "https://youtu.be/4tASl0auPOg?si=Ylxs31RQu7-CO9hv",
    "https://youtu.be/tYKMPVWBaTU?si=z0EAeEsIutu0yt_P",
    "https://youtu.be/mik3IMN9als?si=X6Yx_7UfazI5Q1-w",
    "https://youtu.be/xT6izBdlHHI?si=CqKlMH2A85TGjLnd",
    "https://youtu.be/3yY4ai2L2BA?si=flDWxnLUbCbT2w6f",
    "https://youtu.be/Dq-d_5_G4FA?si=dv97kun8dW6AMyOK",
    "https://youtu.be/OngTe0aPU_E?si=oTzi_9FXypHpQ7PW",
    "https://youtu.be/DS40QYeaOOs?si=i-D520z3xMTCZXE3",
    "https://youtu.be/P5o_jwQEiEQ?si=lvqSZkbmpIs3bJfS",
    "https://youtu.be/B_vyn_PhUCg?si=vCDfrjVAkgUKLuQv",
    "https://youtu.be/bdkrU4ztCbk?si=s51Zo20YyfxUMYm-",
    "https://youtu.be/3yLvNTYEUOU?si=uPnme-0x1DE42bCe",
    "https://youtu.be/jWDL0iP_wnA?si=A9vnCkBdgECHANQu",
    "https://youtu.be/6bnylNUA6Sc?si=2Nk7aNUBPr0uQ1l7",
    "https://youtu.be/8LPVjHxXvJM?si=ovuBAXQybGhXgSz5",
    "https://youtu.be/MlQsm_HCuKc?si=O71vponQrgVEk2Wq",
    "https://youtube.com/shorts/vLoPG3fniQw?si=yrtkAggT1IprnR73",
    "https://youtube.com/shorts/DALbG77IwCM?si=U5Lv7sOQ0hUG3Asl",
    "https://youtu.be/s5bI_732Cqs?si=wGsQ3q96FQUBVxQM",
    "https://youtu.be/LX2dDI1KkQ8?si=AYCU29bm5Bc3rN8A",
    "https://youtu.be/bAb8KIhgVAI?si=0KgDbaXOOdnj1Cpb",
    "https://youtu.be/Vjm54vQjnSA?si=_GMB0CEIBgTN6U-z",
    "https://youtu.be/2npffg7lJXI?si=G3DpSNXzZAy1n1Tn",
    "https://youtu.be/a4uB7kAUxxo?si=gBuikMtSQW6h80bn",
    "https://youtu.be/5WI6MHH7Y7I?si=QzRuemgDxE9scMg9",
    "https://youtu.be/uC-wPiAZnnU?si=GnB1p7NGOQWbCo8_",
    "https://youtu.be/uC-wPiAZnnU?si=KodP9s6oI3cFhLkQ",
    "https://youtu.be/FNLZzqn2EtY?si=mbd4zFnFKmi9bnX1",
    "https://youtu.be/P0KTcOr8Jjg?si=zHUS2P6vlutC_0nX"
]

# Make folders
os.makedirs("data/test", exist_ok=True)

metadata = {}
skipped_urls = []


# Download function
def download_and_log(urls, label):
    for url in urls:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': f"data/test/%(id)s.%(ext)s",
            # Remove the cookiesfrombrowser option since you want to skip these videos
            # 'cookiesfrombrowser': 'chrome',
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = f"{info['id']}.mp4"
                metadata[filename] = {
                    'filename': filename,
                    'label': label,
                    'split':'test'
                }
        except yt_dlp.utils.DownloadError:
            # Catch the error specifically for download issues
            print(f"⚠️ Skipping video due to download error: {url}")
            skipped_urls.append(url)
        except Exception as e:
            # Catch any other unexpected errors
            print(f"❌ An unexpected error occurred with {url}: {e}")
            skipped_urls.append(url)

# Download
download_and_log(deepfake_urls, "FAKE")
download_and_log(real_urls, "REAL")

# Save metadata
# with open("data/test/metadata.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=["filename", "label", "source", "url"])
#     writer.writeheader()
#     writer.writerows(metadata.values())
with open("data/test/metadata.json", "w", encoding="utf-8") as f:json.dump(metadata, f, indent=4)

print("\n--- Summary ---")
print(f"✅ Download complete. Metadata for {len(metadata)} videos saved to dataset/metadata.csv")
if skipped_urls:
    print(f"❌ Skipped {len(skipped_urls)} videos.")
    print("Skipped URLs:")
    for url in skipped_urls:
        print(f"  - {url}")