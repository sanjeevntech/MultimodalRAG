from sentence_transformers import SentenceTransformer
import chromadb
from PIL import Image
import matplotlib.pyplot as plt   # ✅ Import matplotlib here

# 1. Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# 2. Initialize ChromaDB
#client = chromadb.Client()
#collection = client.create_collection("image_rag_demo")

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
    image = Image.open(img_path)
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
        img = Image.open(doc)
        plt.subplot(1, len(retrieved_docs), i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(doc)
    plt.suptitle(f"Query: {q}")
    plt.show()