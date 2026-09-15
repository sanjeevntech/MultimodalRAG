import os
import time
from sentence_transformers import SentenceTransformer
import chromadb
from PIL import Image
import matplotlib.pyplot as plt   # ✅ Import matplotlib here

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# 2. Initialize ChromaDB
# Create or connect to a persistent database
client = chromadb.PersistentClient(path="D:/VSCODE/PythonSamples/MULTIMODALRAG/chroma_store")

# If collection already exists, get it; otherwise create it
try:
    collection = client.get_collection("image_rag_demo")
except:
    collection = client.create_collection("image_rag_demo")


# 3. Add sample images
images = ["c1.jpg", "c2.jpg", "c3.jpg", "d1.jpg", "d2.jpg", "d3.jpg"]
for idx, img_path in enumerate(images):
    image = Image.open(os.path.join(SCRIPT_DIR, img_path))
    embedding = model.encode(image)
    collection.add(
        documents=[img_path],  # store filename as metadata
        embeddings=[embedding.tolist()],
        ids=[f"img{idx}"]
    )
    

print("✅ Images added to ChromaDB")

queries = [ 
    "find the sleeping cat image",
    "find the black cat image which is single",
    "find siberian husky",
    "find dalmatian",
    "find german shepherd",
    "find image having two cats"
]
'''
for q in queries:
    print(f"\n🔍 Query: {q}")
    q_embedding = model.encode(q)
    results = collection.query(
        query_embeddings=[q_embedding.tolist()],
        n_results=1
    )
    retrieved_docs = results["documents"][0]
    print("Results:", retrieved_docs)

    # Plot results for this query
    plt.figure(figsize=(8,4))
    for i, doc in enumerate(retrieved_docs):
        img = Image.open(os.path.join(SCRIPT_DIR, doc))
        plt.subplot(1, len(retrieved_docs), i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(doc)
    plt.suptitle(f"Query: {q}")
    plt.show()
    time.sleep(1)
'''
# 4. Image-to-image search: use one of the sample images as the query
query_image_path = "dogsearch.jpg"
query_image = Image.open(os.path.join(SCRIPT_DIR, query_image_path))
query_image_embedding = model.encode(query_image)

image_results = collection.query(
    query_embeddings=[query_image_embedding.tolist()],
    n_results=1  # top match will usually be the image itself, so ask for 2
)
retrieved_docs = [
    doc for doc in image_results["documents"][0] if doc != query_image_path
] or image_results["documents"][0]

print(f"\n🖼️ Image query: {query_image_path}")
print("Results:", retrieved_docs)

plt.figure(figsize=(8, 4))
plt.subplot(1, len(retrieved_docs) + 1, 1)
plt.imshow(query_image)
plt.axis("off")
plt.title(f"Query: {query_image_path}")
for i, doc in enumerate(retrieved_docs):
    img = Image.open(os.path.join(SCRIPT_DIR, doc))
    plt.subplot(1, len(retrieved_docs) + 1, i + 2)
    plt.imshow(img)
    plt.axis("off")
    plt.title(doc)
plt.suptitle("Image-to-image search")
plt.show()