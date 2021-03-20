import pymongo

def con():
	client = pymongo.MongoClient()

	mydb = client["intents"]

	intents_col = mydb["intents"]

	return list(intents_col.find())