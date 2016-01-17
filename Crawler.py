from bs4 import BeautifulSoup
from pprint import pprint
import requests
import json

MAX_INIT_DOCS = 10
MAX_DOCS = 1000
all_doc_ids = [] # not removed
links = []
json_docs = []

url = "http://www.researchgate.net/researcher/8159937_Zoubin_Ghahramani"
page = requests.get(url)

soup = BeautifulSoup(page.text, "html.parser")

data_link = soup.find_all('a', attrs={'class': 'js-publication-title-link ga-publication-item'})[:10]
data_title = soup.find_all('span', attrs={'class': 'publication-title js-publication-title'})[:10]
data_abstract = soup.find_all('span', attrs={'class': 'full'})[:10]
data_authors = soup.find_all('div', attrs={'class': 'authors'})[:10]

for i in range(10):
    links.append(data_link[i].get('href'))
    json_item = {}
    json_item['title'] = data_title[i].text
    json_item['id'] = data_link[i].get('href')[12:21]
    json_item['abstract'] = data_abstract[i].text.replace('[Hide abstract] ABSTRACT:', '')

    my_authors = []
    for a in data_authors[i].find_all('a'):
        my_authors.append(a.text)
    my_authors.append('Zoubin Ghahramani')
    json_item['authors'] = my_authors

    json_docs.append(json_item)

#for x in json_docs:
    #print(x)

IDs = []

counter = 0
while counter < MAX_INIT_DOCS:
    all_doc_ids.append(links[0][12:21])
    cite_receiver = 'https://www.researchgate.net/publicliterature.PublicationIncomingCitationsList.html?publicationUid='+links[0][12:21]+'&usePlainButton=false&showEnrichedPublicationItem=true&useEnrichedContext=true&showAbstract=false&showType=false&showDownloadButton=false&showOpenReviewButton=false&showPublicationPreview=false&swapJournalAndAuthorPositions=true'
    ref_receiver = 'https://www.researchgate.net/publicliterature.PublicationCitationsList.html?publicationUid='+links[0][12:21]+'&usePlainButton=false&showEnrichedPublicationItem=true&showAbstract=false&showType=false&showDownloadButton=false&showOpenReviewButton=false&showPublicationPreview=false&swapJournalAndAuthorPositions=true'
    payload = {'accept': 'application/json', 'x-requested-with': 'XMLHttpRequest'}
    cite_req = requests.get(cite_receiver, headers=payload)
    ref_req = requests.get(ref_receiver, headers=payload)
    links.pop(0)

    # retrieve references
    objs = json.loads(ref_req.text)['result']['data']['citationItems']
    my_refs = []
    for article in objs:
        if not article['data']['publicationUid'] in all_doc_ids:
            IDs.append(article['data']['publicationUid'])
            all_doc_ids.append(article['data']['publicationUid'])
            my_refs.append(article['data']['publicationUid'])
    json_docs.


    # retrieve citations
    objs = json.loads(cite_req.text)['result']['data']['citationItems']
    for article in objs:
        if not article['data']['publicationUid'] in all_doc_ids:
            IDs.append(article['data']['publicationUid'])
            all_doc_ids.append(article['data']['publicationUid'])

    # id, name, abstract, authors, cites, refs
    counter = counter + 1

#while len(all_doc_ids) < MAX_DOCS:
