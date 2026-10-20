import argparse
from pathlib import Path

import pandas as pd


def read_csv(data_path: Path) -> pd.DataFrame:
    # Trying , and ; as delimiters.
    try:
        df = pd.read_csv(data_path, index_col=0)
    except:
        try:
            df = pd.read_csv(data_path, delimiter=";", index_col=0)
        except Exception as exc:
            raise RuntimeError(exc)
    # Hack for user study csvs, remove NaN in the "id" column (there are explanation cells).
    # if id in df.columns:
    #     df = df[~df.id.isna()]
    return df


def get_concept_score(data_path: Path, key, concept, l_list, pid=False):
    df = read_csv(data_path)

    # print(data_path)
    # print("WASDASDSADASDS")
    scores = []
    for l in l_list:
        print(f"lambda: {l}")
        responses = df.loc[df["lambda"] == l, f"{concept}_q0_llm_answer"].tolist()
        print(responses)
        score = sum([r == 'Yes' for r in  responses])/len(responses)
        scores.append(score)
        print(f"score: {score}")

    results_df = pd.DataFrame(
            data=scores,
            columns=[  # cfg.columns +
                f"{concept}_score",
            ]
        )
    dfs_out = (
        [
            df,
        ]
    )
    results_final = pd.concat(
        dfs_out + [results_df],
        axis=1,
    )
    output_path = "./concepts/" + key + "/"
    
    if output_path is not None:
        if pid:
            filename = Path(output_path) / ('PID' + concept + "_score_eval.csv")
            # output_path = output_path + 'PID'
        else:
            # filename = Path(output_path) / (concept + "_score_eval.csv")
            filename = data_path
        results_final.to_csv(filename)
        print(f"Saved results in {filename}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate concept scores from an evaluation CSV."
    )
    parser.add_argument("data_path", type=Path, help="path to the evaluation CSV")
    parser.add_argument("concept", help="concept name used in the *_q0_llm_answer column")
    parser.add_argument(
        "lambdas",
        type=float,
        nargs="+",
        help="lambda values to score",
    )
    args = parser.parse_args()

    get_concept_score(
        data_path=args.data_path,
        key="",
        concept=args.concept,
        l_list=args.lambdas,
    )

if __name__ == "__main__":
    main()