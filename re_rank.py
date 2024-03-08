
from sentence_transformers import SentenceTransformer, util ,CrossEncoder



# Load the model, here we use our base sized model
model = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")


# Example query and documents




def  test_model(query):
#	query = "who provide policies"
	documents = ["vManage, vBond, and vSmart are components of Cisco's SD-WAN (Software-Defined Wide Area Network) solution.", 'vManage is the centralized management and orchestration platform for the SD-WAN solution. It provides a single pane of glass to manage and monitor the entire network infrastructure, including physical and virtual devices, applications, policies, and security.', 'vBond is the orchestrator responsible for establishing secure and encrypted control-plane connectivity between SD-WAN devices, including vSmart controllers and vEdge routers. It provides a secure method of establishing and maintaining a trust relationship between these devices.', 'Note: The vBond orchestrator automatically authenticates all other SD-WAN devices when they join the SD-WAN overlay network.', '', 'vSmart is the brains of the SD-WAN solution, providing policy-based control-plane connectivity between SD-WAN devices. It uses software-defined intelligence to determine the best path for traffic based on business policies, network conditions, and application requirements. It also provides end-to-end network visibility, analytics, and reporting.', "The vManage Network Management System (NMS) is the central management and orchestration platform for Cisco's SD-WAN solution. It provides a simple yet powerful set of graphical dashboards for monitoring network performance on all devices in the overlay network from a centralized monitoring station.", 'In addition to network monitoring, the vManage NMS also provides centralized software installation, upgrade, and provisioning. This includes the ability to perform these operations for a single device or as a bulk operation for many devices simultaneously. This centralized approach to software management simplifies the overall management of the network and reduces the potential for errors or inconsistencies that can arise when managing software on multiple devices.']
# Calculate the scores
	results = model.rank(query, documents, return_documents=True, top_k=3)
	print(results)

while True:
	query = input("\033[32mPrompt>\033[0m")
	#query = "How many people live in London?"
# Assuming 'docs' is a list of sentences/chunks from your 200 lines of text
	if query:
		test_model(query)
	else:
		break