# face recognizing website backend

from fastapi import FastAPI
from flask import session
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Identity
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Session
from fastapi import Depends
from fastapi import UploadFile, File
from sqlalchemy import text
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS configuration to allow requests from the frontend
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()

# Load DATABASE_URL from .env file
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file. Please check your .env file.")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, Identity(start=1, increment=1), primary_key=True)
    full_name = Column(String)
    email = Column(String)
    address1 = Column(String)
    address2 = Column(String)
    phone = Column(String)

    # InsightFace usually outputs 512-dim vectors
    embedding = Column(Vector(512))

Base.metadata.create_all(bind=engine)

face_analysis = FaceAnalysis(name='buffalo_l', root='./models')
face_analysis.prepare(ctx_id=0, det_size=(640, 640))


@app.post("/recognize-face")
async def recognize_face(frame: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await frame.read()

    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    faces = face_analysis.get(image)

    if len(faces) == 0:
        return {"message": "No face detected in the image."}
    
    feat2 = faces[0].embedding

    feat2_norm = feat2 / np.linalg.norm(feat2)

    embedding_str = "[" + ",".join(map(str, feat2_norm.tolist())) + "]"

    result = db.execute(
        text("""
            SELECT id, full_name, email,
                   embedding <=> :embedding AS distance
            FROM users
            ORDER BY embedding <=> :embedding
            LIMIT 1
        """),
        {"embedding": embedding_str}
    )
    user = result.fetchone()

    if user and user.distance < 0.3:
        return {
            "status": "approved",
            "recognized": True,
            "name": user.full_name,
            "email": user.email,
            "distance": float(user.distance)
        }

    return {"status": "rejected", "recognized": False}



class RegisterRequest(BaseModel):
    fullName: str
    email: str
    address1: str
    address2: str
    phone: str
    photo: str

@app.post("/register")
async def register_face(data: RegisterRequest, db: Session = Depends(get_db)):
    fullName = data.fullName
    email = data.email
    address1 = data.address1
    address2 = data.address2
    phone = data.phone
    photo = data.photo

    # Decode the base64 image

    base64_data = data.photo.split(",")[1]

    image_bytes = base64.b64decode(base64_data)

    nparr = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = face_analysis.get(image)
    rimg = face_analysis.draw_on(image, faces)
    rimg_rgb = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)

    # Display the image with detected faces (for testing purposes)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(rimg_rgb)
    # plt.axis('off')
    # plt.title('Face Detection with insightface (buffalo_l)')
    # plt.show()

    feat1 = faces[0].embedding

    user = User(

        full_name=fullName,
        email=email,
        address1=address1,
        address2=address2,
        phone=phone,
        embedding=feat1
    )

    db.add(user)
    db.commit()
    db.refresh(user)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)