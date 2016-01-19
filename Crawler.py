import json
import requests

from pprint import pprint
from bs4 import BeautifulSoup


def initialize_queue():
	url = "http://www.researchgate.net/researcher/8159937_Zoubin_Ghahramani"
	initial_page = requests.get(url)

	soup = BeautifulSoup(initial_page.text, "html.parser")

	data_link = soup.find_all('a', attrs={'class': 'js-publication-title-link ga-publication-item'})[:10]
	data_title = soup.find_all('span', attrs={'class': 'publication-title js-publication-title'})[:10]
	data_abstract = soup.find_all('span', attrs={'class': 'full'})[:10]
	data_authors = soup.find_all('div', attrs={'class': 'authors'})[:10]

	initial_json_docs = []

	for i in range(10):
		json_item = {}

		json_item['id'] = int(data_link[i].get('href')[12:21])
		json_item['title'] = data_title[i].text
		json_item['abstract'] = data_abstract[i].text.replace('[Hide abstract] ABSTRACT:', '')

		authors = []
		for a in data_authors[i].find_all('a'):
			authors.append(a.text)
		authors.append('Zoubin Ghahramani')
		json_item['authors'] = authors

		initial_json_docs.append(json_item)

	return initial_json_docs


def extracxt_json_doc(article):
	json_item = {}

	json_item['id'] = article['data']['publicationUid']
	json_item['title'] = article['data']['title']
	json_item['abstract'] = article['data']['abstract']

	authors = []
	for author in article['data']['authors']:
		authors.append(author['fullname'])
	json_item['authors'] = authors

	return json_item


MAX_INIT_DOCS = 10
MAX_DOCS = 1000
all_doc_ids = []

incomplete_json_docs = initialize_queue()
final_json_docs = []

for doc in incomplete_json_docs:
	all_doc_ids.append(doc['id'])

counter = 0
while counter < 20:
	json_doc = incomplete_json_docs.pop(0)
	pUid = json_doc['id']	
	cite_referee = 'https://www.researchgate.net/publicliterature.PublicationIncomingCitationsList.html?publicationUid='+str(pUid)+'&usePlainButton=false&showEnrichedPublicationItem=true&useEnrichedContext=true&showAbstract=true&showType=false&showDownloadButton=false&showOpenReviewButton=false&showPublicationPreview=false&swapJournalAndAuthorPositions=true'
	ref_referee = 'https://www.researchgate.net/publicliterature.PublicationCitationsList.html?publicationUid='+str(pUid)+'&usePlainButton=false&showEnrichedPublicationItem=true&showAbstract=true&showType=false&showDownloadButton=false&showOpenReviewButton=false&showPublicationPreview=false&swapJournalAndAuthorPositions=true'
	headers = {'accept': 'application/json', 'x-requested-with': 'XMLHttpRequest'}
	cite_req = requests.get(cite_referee, headers=headers)
	ref_req = requests.get(ref_referee, headers=headers)

	# retrieve references
	references = json.loads(ref_req.text)['result']['data']['citationItems']
	refs = []
	for article in references:
		refs.append(article['data']['publicationUid'])

		if not article['data']['publicationUid'] in all_doc_ids:
			all_doc_ids.append(article['data']['publicationUid'])
			new_json_doc = extracxt_json_doc(article)
			incomplete_json_docs.append(new_json_doc)

	# retrieve citations
	citations = json.loads(cite_req.text)['result']['data']['citationItems']
	cites = []
	for article in citations:
		cites.append(article['data']['publicationUid'])

		if not article['data']['publicationUid'] in all_doc_ids:
			all_doc_ids.append(article['data']['publicationUid'])
			new_json_doc = extracxt_json_doc(article)
			incomplete_json_docs.append(new_json_doc)

	json_doc['references'] = refs
	json_doc['citations'] = cites
	final_json_docs.append(json_doc)
	counter += 1

for doc in final_json_docs:
	print(doc['id'])
	print(doc['title'])
	print(doc['authors'])
	print(doc['references'])
	print(doc['citations'])
