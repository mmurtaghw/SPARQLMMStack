import pandas as pd
import os
import sys

# constants
data_path = "data/raw/"
llm_stack = { # dictionaries preserve order!
    ("qwen0.5b"): "processed_qald_9_experiment_report_qwen0.5b_none (1).csv",
    ("qwen1.5b"): "processed_qald_9_experiment_report_qwen1.5b_none.csv",
    ("mistral7b"): "processed_qald_9_experiment_report_mistral7b_none.csv",
    ("Gemma2b"): "processed_qald_9_experiment_report_Gemma2b_none.csv",
    ("Gemma9b"): "processed_qald_9_experiment_report_Gemma9b_none (1).csv"
}
cuttoffs = {
    ("qwen0.5b"): 0.0,
    ("qwen1.5b"): 0.0,
    ("mistral7b"): 0.0,
    ("Gemma2b"): 0.0,
    ("Gemma9b"): 0.0
}
models = [key for key in llm_stack]

def print_data():
    dfs = {}
    for key in llm_stack:
        path = os.path.join(data_path, llm_stack[key])
        df = pd.read_csv(path)
        question_col = df["question_id"]
        runs_col = df["is_execution_valid"]
        bleu_col = df["bleu_score"].fillna(0)
        bleu_hybrid_col = (runs_col * bleu_col)
        f1_col = df["macro_f1"].fillna(0)
        f1_hybrid_col = (runs_col * f1_col)
        dfs[key] = pd.DataFrame(
            [
                question_col,
                runs_col,
                bleu_col,
                bleu_hybrid_col,
                f1_col,
                f1_hybrid_col
            ]
        )
        dfs[key] = dfs[key].transpose().set_axis(
            [
                "question_col",
                "runs_col",
                "bleu_col",
                "bleu_hybrid_col",
                "f1_col",
                "f1_hybrid_col"
            ], axis=1
        )
    
    for key in dfs:
        print(key)
        dfs[key].to_csv(f"{key}.temp.csv", header=True, index=False)

def load_data():
    dfs = {}
    for key in llm_stack:
        path = os.path.join(data_path, llm_stack[key])
        df = pd.read_csv(path)
        question_col = df["question_id"]
        runs_col = df["is_execution_valid"]

        bleu_col = df["bleu_score"].fillna(0)
        bleu_col_cuttoff = bleu_col > cuttoffs[key]
        bleu_hybrid_col = (runs_col * bleu_col)
        bleu_hybrid_cuttoff = bleu_hybrid_col > cuttoffs[key]

        f1_col = df["macro_f1"].fillna(0)
        f1_col_cuttoff = f1_col > cuttoffs[key] # TODO diff for f1 and bleu
        f1_hybrid_col = (runs_col * f1_col)
        f1_hybrid_cuttoff = f1_hybrid_col > cuttoffs[key]

        df_trunk = pd.DataFrame(
            [
                question_col,
                runs_col,
                bleu_col_cuttoff,
                bleu_hybrid_cuttoff,
                f1_col_cuttoff,
                f1_hybrid_cuttoff
            ]
        ).transpose()
        df_trunk = df_trunk.set_axis(
            [
                "question",
                "is_execution_valid",
                "bleu_col_cuttoff",
                "bleu_hybrid_col_cuttoff",
                "f1_col_cuttoff",
                "f1_hybrid_col_cuttoff"
            ], axis=1
        )
        dfs[key] = df_trunk
    return dfs

def to_levels(dfs, metric):
    '''
    get a mapping from all questions to the first level in the LLM stack that should be used to try to answer them.

    this can be done using various metrics; i.e.
        is_execution_valid
        bleu_col_cuttoff
        hybrid_col_cuttoff
    '''
    # get the level we saw the first success at
    run_rows = list(dfs.values())[0].shape[0]
    levels = []
    for row_idx in range(run_rows):
        for first_success_level, model in enumerate(models):
            df = dfs[model]
            if df.loc[[row_idx]][metric].squeeze():
                break
        levels.append(first_success_level)

    # load the level (and all corresponding Qs) into a dataframe
    levels_col = pd.Series(levels) 
    questions_col = list(dfs.values())[0]["question"]
    df_q_to_level = pd.DataFrame([questions_col, levels_col])
    df_q_to_level = df_q_to_level.transpose()
    df_q_to_level.to_csv("temp.csv", index=False, header=False)

def main():
    print_data()
    # dfs = load_data()
    # to_levels(dfs, "is_execution_valid")

if __name__ == '__main__':
    main()
