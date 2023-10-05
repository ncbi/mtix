import gzip
import json
import os.path
import pandas as pd
import pytrec_eval


WORKING_DIR="/net/intdev/pubmed_mti/ncbi/working_dir/mtix/scripts_v3/create_test_set_mesh_heading_ground_truth"


def create_lookup(path):
    data = pd.read_csv(path, sep="\t", header=None)
    lookup = dict(zip(data.iloc[:,0], data.iloc[:,1]))
    return lookup


def main():
    ground_truth_path =  os.path.join(WORKING_DIR, "val_set_2017-2023_MeSH_Heading_Ground_Truth.json")
    names_path =         os.path.join(WORKING_DIR, "mesh_heading_names_2023.tsv")
    qrels_path =         os.path.join(WORKING_DIR, "val_set_2017-2023_qrels.tsv")
    test_set_data_path = os.path.join(WORKING_DIR, "val_set_2017-2023_data.json.gz")
    types_path =         os.path.join(WORKING_DIR, "mesh_heading_types_2023.tsv")
    uis_path =           os.path.join(WORKING_DIR, "mesh_heading_uis_2023.tsv")

    names = create_lookup(names_path)
    types = create_lookup(types_path)
    uis =   create_lookup(uis_path)
    
    qrels = pytrec_eval.parse_qrel(open(qrels_path))
    test_set_data = json.load(gzip.open(test_set_data_path, mode="rt", encoding="utf8"))
    
    data_lookup = { citation_data["uid"]: citation_data["data"] for citation_data in test_set_data}

    mti_json = []
    for q_id in qrels:
        pmid = int(q_id)
        citation_predictions = { "PMID": pmid, "text-gz-64": data_lookup[pmid],  "Indexing": [] }
        mti_json.append(citation_predictions)
        for p_id in qrels[q_id]: 
            label_id = int(p_id)
            name = names[label_id]
            _type = types[label_id]
            ui =   uis[label_id]
            score = 1.
            citation_predictions["Indexing"].append({
                "Term": name, 
                "Type": _type, 
                "ID": ui, 
                "IM": "NO", 
                "Reason": f"score: {score:.3f}"})

    json.dump(mti_json, open(ground_truth_path, "wt"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()