# pyrefly: ignore [missing-import]
from neuprint import Client, fetch_neurons

client = Client("https://neuprint.janelia.org", dataset='male-cns:v1.0', token='ab15fe59936e000268f2542e0a244091f7a24e4c6797365a6d2974c156cd087a')
neurons, syndist = fetch_neurons("DNge104")
print(neurons.head())
