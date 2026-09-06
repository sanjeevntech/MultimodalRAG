from sentence_transformers import SentenceTransformer
import chromadb
from PIL import Image

# 1. Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# 2. Persistent ChromaDB
client = chromadb.PersistentClient(path="D:/VSCODE/PythonSamples/MULTIMODALRAG/chroma_store")

# 3. Get or create collection
try:
    collection = client.get_collection("image_rag_demo")
except:
    collection = client.create_collection("image_rag_demo")

# 4. Add sample images (only once)
images = ["dog1.jpg", "cat1.jpg", "park.jpg", "car.jpg", "house.jpg", "tree.jpg"]
for idx, img_path in enumerate(images):
    image = Image.open(img_path)
    embedding = model.encode(image)
    collection.add(
        documents=[img_path],
        embeddings=[embedding.tolist()],
        ids=[f"img{idx}"]
    )

print("✅ Images added to ChromaDB")

# 5. Query with an image (instead of text)
query_image = Image.open("dog1.jpg")   # pick one of your images
query_embedding = model.encode(query_image)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=2
)

retrieved_docs = results["documents"][0]
print("🔍 Image-to-image search results:", retrieved_docs)

# 6. Display retrieved images
for doc in retrieved_docs:
    img = Image.open(doc)
    img.show()   # Opens in default image viewer
