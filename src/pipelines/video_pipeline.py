import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms


mp_face = mp.solutions.face_mesh


class VideoPipeline:
def __init__(self, model_path=None, device='cpu'):
self.device = device
self.face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
# placeholder embedding model - small resnet
self.embedding_model = None
if model_path:
self.embedding_model = torch.load(model_path, map_location=device)
self.embedding_model.eval()
self.transform = transforms.Compose([
transforms.ToPILImage(),
transforms.Resize((112,112)),
transforms.ToTensor(),
])


def detect_face(self, frame):
# returns cropped face or None
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = self.face_mesh.process(rgb)
if not results.multi_face_landmarks:
return None
# compute bounding box from landmarks
h, w, _ = frame.shape
xs = [lm.x for lm in results.multi_face_landmarks[0].landmark]
ys = [lm.y for lm in results.multi_face_landmarks[0].landmark]
minx, maxx = int(min(xs)*w), int(max(xs)*w)
miny, maxy = int(min(ys)*h), int(max(ys)*h)
# add padding
pad = 10
minx = max(0, minx-pad); miny = max(0, miny-pad)
maxx = min(w, maxx+pad); maxy = min(h, maxy+pad)
face = frame[miny:maxy, minx:maxx]
return face


def get_embedding(self, face_img):
if face_img is None:
return None
x = self.transform(face_img).unsqueeze(0).to(self.device)
if self.embedding_model is None:
# fallback simple feature: mean pixels
return torch.tensor(x.mean()).unsqueeze(0)
with torch.no_grad():
emb = self.embedding_model(x)
}
