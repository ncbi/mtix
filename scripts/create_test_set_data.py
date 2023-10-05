import base64
import gzip
import json
import os.path
import xml.etree.ElementTree as ET
import zlib


BASELINE_FILE_PATH_TEMPLATE="/net/intdev/pubmed_mti/ncbi/working_dir/mtix/mh_scr_pt_prediction/medline_data_2023/pubmed23n{0:04d}.xml.gz"
ENCODING="utf-8"
NUM_BASELINE_FILES = 1166
WORKING_DIR="/net/intdev/pubmed_mti/ncbi/working_dir/mtix/scripts_v3/create_test_set_data"


def base64_encode(text, level=-1):
    text = text.encode(ENCODING)
    text = zlib.compress(text, level=level)
    text = base64.b64encode(text) 
    text = text.decode(ENCODING) 
    return text


def get_test_set_pmids(test_set_path):
    pmid_list = None
    with gzip.open(test_set_path, "rt", encoding=ENCODING) as file:
        pmid_list = [int(json.loads(line.strip())["pmid"]) for line in file.readlines()]
    return pmid_list
            

def get_test_set_data(test_set_pmids):
    data = {}
    test_set_pmids_set = set(test_set_pmids)
    for file_num in range(1, NUM_BASELINE_FILES + 1):
        print(f"{file_num}/{NUM_BASELINE_FILES}", end="\r")
        file_path = BASELINE_FILE_PATH_TEMPLATE.format(file_num)
        with gzip.open(file_path, "rt", encoding=ENCODING) as read_file:
            root_node = ET.parse(read_file)
            for medline_citation_node in root_node.findall("PubmedArticle/MedlineCitation"):
                pmid = int(medline_citation_node.find("PMID").text.strip())
                if pmid in test_set_pmids_set:
                    medline_citation_node_xml = ET.tostring(medline_citation_node, encoding=ENCODING, method="xml").decode(ENCODING)
                    medline_citation_node_xml = base64_encode(medline_citation_node_xml)
                    data[pmid] = json.dumps({ "uid": pmid, "data": medline_citation_node_xml }, ensure_ascii=False) 
    return data


def save_test_set_data(pmid_list, data, path):
    with open(path, "wt", encoding=ENCODING) as write_file:
        first = True
        for pmid in pmid_list:
            if pmid in data:
                if first:
                    write_file.write("[")
                    first = False
                else:
                    write_file.write(",\n")
                write_file.write(data[pmid])
        write_file.write("]")
                

def main():
    test_set_path =      os.path.join(WORKING_DIR, "val_set_2017-2023.jsonl.gz")
    test_set_data_path = os.path.join(WORKING_DIR, "val_set_2017-2023_data.json")
    pmids_list = get_test_set_pmids(test_set_path)
    data = get_test_set_data(pmids_list)
    save_test_set_data(pmids_list, data, test_set_data_path)
    
    
if __name__ == "__main__":
    main()