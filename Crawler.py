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


def save_json_to_file(json_item, filename_convention):
    with open(str(json_item[filename_convention]), 'w') as f:
        json.dump(json_item, f, ensure_ascii=False)


def log_frontier_queue(f):
    f = open('frontier', 'r+')
    f.seek(0)
    # f.write(incomplete_json_docs)

    for item in incomplete_json_docs:
        # f.write("%s\n" % item)
        json.dump(item, f)

    f.truncate()
    f.close()


# TODO: maybe we need to implement this in future
def load_queue_from_log():
    pass


MAX_INIT_DOCS = 10
MAX_DOCS = 12
all_doc_ids = []

# create the file for logging the frontier
log_frontier = open('frontier', 'w')
log_frontier.close()

incomplete_json_docs = initialize_queue()
final_json_docs = []

for doc in incomplete_json_docs:
    all_doc_ids.append(doc['id'])

counter = 0
while len(final_json_docs) < MAX_DOCS:
    try:
        print('I am working fine...' + str(len(final_json_docs)))
        json_doc = incomplete_json_docs.pop(0)
        pUid = json_doc['id']
        cite_referee = 'https://www.researchgate.net/publicliterature.PublicationIncomingCitationsList.html?publicationUid=' + str(
            pUid) + '&usePlainButton=false&showEnrichedPublicationItem=true&useEnrichedContext=true&showAbstract=true&showType=false&showDownloadButton=false&showOpenReviewButton=false&showPublicationPreview=false&swapJournalAndAuthorPositions=true'
        ref_referee = 'https://www.researchgate.net/publicliterature.PublicationCitationsList.html?publicationUid=' + str(
            pUid) + '&usePlainButton=false&showEnrichedPublicationItem=true&showAbstract=true&showType=false&showDownloadButton=false&showOpenReviewButton=false&showPublicationPreview=false&swapJournalAndAuthorPositions=true'
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

        if len(citations) > 0:
            more_cite_referee = 'https://www.researchgate.net/publicliterature.PublicationIncomingCitationsList.html?publicationUid=' + str(
                pUid) + '&usePlainButton=0&useEnrichedContext=1&swapJournalAndAuthorPositions=1&showAbstract=1&showOpenReviewButton=0&showDownloadButton=0&showType=0&showPublicationPreview=0&showEnrichedPublicationItem=1&publicationUid=' + str(
                pUid) + '&limit=10&offset=3'
            headers = {'accept': 'application/json', 'x-requested-with': 'XMLHttpRequest'}
            more_cite_req = requests.get(more_cite_referee, headers=headers)

            citations.extend(json.loads(more_cite_req.text)['result']['data']['citationItems'][:10 - len(citations)])

            for article in citations:
                cites.append(article['data']['publicationUid'])

                if not article['data']['publicationUid'] in all_doc_ids:
                    all_doc_ids.append(article['data']['publicationUid'])
                    new_json_doc = extracxt_json_doc(article)
                    incomplete_json_docs.append(new_json_doc)

        json_doc['references'] = refs
        json_doc['citations'] = cites
        final_json_docs.append(json_doc)

        # save each completed json to a file
        save_json_to_file(json_doc, 'id')

        # save the current frontier for the catastrophic case!
        # every 10 json_item
        if counter % 10 == 0:
            log_frontier_queue(log_frontier)

        counter += 1
    # if connection was not OK, continue the loop, do not panic!
    except ConnectionError:
        print('I am idle -- waiting for connection...')

#for doc in final_json_docs:
#    print(doc['id'])
#    print(doc['title'])
#    print(doc['authors'])
#    print(doc['references'])
#    print(doc['citations'])
