
#"score: 0.995"

import json


def update_format(indexed_term):
    reason = indexed_term["Reason"]
    score = reason.split(":")[1].strip()
    score = float(score)
    score = round(score, 3)
    indexed_term["Reason"] = "ML Score"
    indexed_term["Score"] = score


def main():
    INPUT_PATH  = "/net/intdev/pubmed_mti/ncbi/working_dir/mtix/scripts_v3/data_old_format/val_set_2017-2023_Subheading_Predictions.json"
    OUTPUT_PATH = "/net/intdev/pubmed_mti/ncbi/working_dir/mtix/scripts_v3/data_new_format/val_set_2017-2023_Subheading_Predictions.json"

    data = json.load(open(INPUT_PATH))
    for citation_prediction in data:
        for indexed_mh in citation_prediction["Indexing"]:
            update_format(indexed_mh)
            if "Subheadings" in indexed_mh:
                for indexed_sh in indexed_mh["Subheadings"]:
                    update_format(indexed_sh)

    json.dump(data, open(OUTPUT_PATH, "wt"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()