import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# constants
data_path = "data/raw/"
def load_qald_9_data():
    global llm_stack, bleu_cuttoffs, f1_cuttoffs, models
    llm_stack = { # dictionaries preserve order!
        ("qwen0.5b"): "processed_qald_9_experiment_report_qwen0.5b_none (1).csv",
        ("qwen1.5b"): "processed_qald_9_experiment_report_qwen1.5b_none.csv",
        ("mistral7b"): "processed_qald_9_experiment_report_mistral7b_none.csv",
        ("Gemma2b"): "processed_qald_9_experiment_report_Gemma2b_none.csv",
        ("Gemma9b"): "processed_qald_9_experiment_report_Gemma9b_none (1).csv"
    }
    bleu_cuttoffs = {
        ("qwen0.5b"): 0.01,
        ("qwen1.5b"): 0.01,
        ("mistral7b"): 0.01,
        ("Gemma2b"): 0.01,
        ("Gemma9b"): 0.01
    }
    f1_cuttoffs = { # one st dev below the mean
        ("qwen0.5b"): 0.09445996094,
        ("qwen1.5b"): 0.1233251932,
        ("mistral7b"): 0.1261542371,
        ("Gemma2b"): 0.1337644674,
        ("Gemma9b"): 0.1605610769
    }
    models = [key for key in llm_stack]

def load_quanda_data():
    raise NotImplementedError
    global llm_stack, bleu_cuttoffs, f1_cuttoffs, models
    llm_stack = { # dictionaries preserve order!
        ("qwen0.5b"): "",
        ("qwen1.5b"): "",
        ("mistral7b"): "",
        ("Gemma2b"): "",
        ("Gemma9b"): ""
    }
    bleu_cuttoffs = {
        ("qwen0.5b"): 0.00,
        ("qwen1.5b"): 0.00,
        ("mistral7b"): 0.00,
        ("Gemma2b"): 0.00,
        ("Gemma9b"): 0.00
    }
    f1_cuttoffs = { # one st dev below the mean
        ("qwen0.5b"): 0.00,
        ("qwen1.5b"): 0.00,
        ("mistral7b"): 0.00,
        ("Gemma2b"): 0.00,
        ("Gemma9b"): 0.00
    }
    models = [key for key in llm_stack]

def print_data(tag):
    dfs = {}
    for key in llm_stack:
        path = os.path.join(data_path, llm_stack[key])
        df = pd.read_csv(path)
        df = df.sort_values(by=['question_id']) # sort so we have consistent order in all cases
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
        dfs[key].to_csv(f"{key}.{tag}.csv", header=True, index=False)

def load_data():
    dfs = {}
    for key in llm_stack:
        path = os.path.join(data_path, llm_stack[key])
        df = pd.read_csv(path)
        df = df.sort_values(by=['question_id']) # sort so we have consistent order in all cases
        question_col = df["question"]
        runs_col = df["is_execution_valid"]

        bleu_col = df["bleu_score"].fillna(0)
        bleu_col_cuttoff = bleu_col > bleu_cuttoffs[key]
        bleu_hybrid_col = (runs_col * bleu_col)
        bleu_hybrid_cuttoff = bleu_hybrid_col > bleu_cuttoffs[key]

        f1_col = df["macro_f1"].fillna(0)
        f1_col_cuttoff = f1_col > f1_cuttoffs[key]
        f1_hybrid_col = (runs_col * f1_col)
        f1_hybrid_cuttoff = f1_hybrid_col > f1_cuttoffs[key]

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
    
    # train test split
    df_train, df_test = train_test_split(df_q_to_level, test_size=0.20, random_state=42)
    df_train.to_csv(f"roberta_data.{metric}.train.csv", index=False, header=False)
    df_test.to_csv(f"roberta_data.{metric}.test.csv", index=False, header=False)

def main():
    metric = sys.argv[1]
    # possible metrics:
    '''
    "is_execution_valid",
    "bleu_col_cuttoff",
    "bleu_hybrid_col_cuttoff",
    "f1_col_cuttoff",
    "f1_hybrid_col_cuttoff"
    '''

    data_src = sys.argv[2] # qald9 or quanda
    if data_src == "qald9":
        load_qald_9_data()
    elif data_src == "quanda":
        load_quanda_data()
    else:
        assert False, f"unknown data src given: {data_src}"

    dfs = load_data()
    to_levels(dfs, metric)

if __name__ == '__main__':
    main()
