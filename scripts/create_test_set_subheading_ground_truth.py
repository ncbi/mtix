import copy
import json
import os.path
import pandas as pd


WORKING_DIR="/net/intdev/pubmed_mti/ncbi/working_dir/mtix/scripts_v3/create_test_set_subheading_ground_truth"


def create_lookup(path):
    data = pd.read_csv(path, sep="\t", header=None)
    lookup = dict(zip(data.iloc[:,0], data.iloc[:,1]))
    return lookup


def get_test_set_indexing(test_set_path):
    test_set_indexing = {}
    for line in open(test_set_path):
        citation_data = json.loads(line.strip())
        pmid = int(citation_data["pmid"])
        test_set_indexing[pmid] = {}
        for dui, qui_list in citation_data["mesh_headings"]:
            test_set_indexing[pmid][dui] = qui_list
    return test_set_indexing


def main():
    mesh_heading_ground_truth_path = os.path.join(WORKING_DIR, "val_set_2017-2023_MeSH_Heading_Ground_Truth.json")
    subheading_ground_truth_path =   os.path.join(WORKING_DIR, "val_set_2017-2023_Subheading_Ground_Truth.json")
    subheading_names_path =          os.path.join(WORKING_DIR, "subheading_names_2023_mesh.tsv")
    test_set_path =                  os.path.join(WORKING_DIR, "val_set.jsonl")

    subheading_names = create_lookup(subheading_names_path)
    mesh_heading_ground_truth = json.load(open(mesh_heading_ground_truth_path))
    mesh_heading_ground_truth_mod = copy.deepcopy(mesh_heading_ground_truth)
    
    test_set_indexing = get_test_set_indexing(test_set_path)

    for example in mesh_heading_ground_truth_mod:
        pmid = int(example["PMID"])
        for example_indexing in example["Indexing"]:
            if example_indexing["Type"].lower() ==  "descriptor" or example_indexing["Type"].lower() ==  "check tag":
                example_indexing["Subheadings"] = []
                dui = example_indexing["ID"]
                if pmid in test_set_indexing and dui in test_set_indexing[pmid]:
                    for qui in test_set_indexing[pmid][dui]:
                        example_indexing["Subheadings"].append({
                            "ID": qui,
                            "IM": "NO",
                            "Name": subheading_names[qui],
                            "Reason": f"score: {1.:.3f}"})

    json.dump(mesh_heading_ground_truth_mod, open(subheading_ground_truth_path, "wt"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()