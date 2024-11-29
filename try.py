import os
import os.path as osp
import json
import subprocess
import modal


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CORE_API_KEY = os.environ.get("CORE_API_KEY")

dockerfile_image = (
    modal.Image.from_dockerfile("Dockerfile")
    .pip_install("pydantic==2.9.2")
    .env(
        {
            "OPENAI_API_KEY": OPENAI_API_KEY,
            "CORE_API_KEY": CORE_API_KEY,
        }
    )
)

app = modal.App("scientist")
vol = modal.Volume.from_name("scientist-vol")


@app.function(gpu=modal.gpu.A10G(), image=dockerfile_image)
def create_img():
    import argparse
    import json
    import multiprocessing
    import openai
    import os
    import os.path as osp
    import shutil
    import sys
    import time
    import torch
    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model
    from datetime import datetime

    from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
    from ai_scientist.llm import create_client, AVAILABLE_LLMS
    from ai_scientist.perform_experiments import perform_experiments
    from ai_scientist.perform_review import (
        perform_review,
        load_paper,
        perform_improvement,
    )
    from ai_scientist.perform_writeup import perform_writeup, generate_latex

    print("all good")


@app.function(gpu=modal.gpu.A10G(), image=dockerfile_image)
def test_imports():
    import argparse
    import json
    import multiprocessing
    import openai
    import os
    import os.path as osp
    import shutil
    import sys
    import time
    import torch
    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model
    from datetime import datetime

    from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
    from ai_scientist.llm import create_client, AVAILABLE_LLMS
    from ai_scientist.perform_experiments import perform_experiments
    from ai_scientist.perform_review import (
        perform_review,
        load_paper,
        perform_improvement,
    )
    from ai_scientist.perform_writeup import perform_writeup, generate_latex

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    CORE_API_KEY = os.environ.get("CORE_API_KEY")
    print("all imports are good to go")
    print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
    print(f"CORE_API_KEY: {CORE_API_KEY}")


@app.function(gpu=modal.gpu.A10G(), image=dockerfile_image)
def get_research_ideas(base_dir, client_model, check_novelty=False):
    import argparse
    import json
    import multiprocessing
    import openai
    import os
    import os.path as osp
    import shutil
    import sys
    import time
    import torch
    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model
    from datetime import datetime

    from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
    from ai_scientist.llm import create_client, AVAILABLE_LLMS
    from ai_scientist.perform_experiments import perform_experiments
    from ai_scientist.perform_review import (
        perform_review,
        load_paper,
        perform_improvement,
    )
    from ai_scientist.perform_writeup import perform_writeup, generate_latex

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=False,
        max_num_generations=2,
        num_reflections=3,
    )

    if check_novelty:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )

    return ideas


@app.function(
    gpu="A100", image=dockerfile_image, volumes={"/data": vol}, timeout=3600 * 4
)
def do_idea(
    base_dir,
    results_dir,
    idea,
    model,
    client_model,
    writeup,
    improvement,
    log_file=False,
):
    import argparse
    import json
    import multiprocessing
    import openai
    import os
    import os.path as osp
    import shutil
    import sys
    import time
    import torch
    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model
    from datetime import datetime

    from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
    from ai_scientist.llm import create_client, AVAILABLE_LLMS
    from ai_scientist.perform_experiments import perform_experiments
    from ai_scientist.perform_review import (
        perform_review,
        load_paper,
        perform_improvement,
    )
    from ai_scientist.perform_writeup import perform_writeup, generate_latex

    vol_base_dir = f"/data/{base_dir}"
    vol_results_dir = f"/data/{results_dir}"

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(vol_results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)

    with open(f"{vol_base_dir}/run_0/final_info.json", "r") as f:
        baseline_results = json.load(f)

    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
        vol.commit()

    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
        vol.commit()

    try:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
            vol.commit()
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"*Starting Writeup*")
        ## PERFORM WRITEUP
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                perform_writeup(
                    idea, folder_name, coder, client, client_model, num_cite_rounds=5
                )
                vol.commit()
            except Exception as e:
                print(f"Failed to perform writeup: {e}")
                return False
            print("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"*Starting Review*")
        ## REVIEW PAPER
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review.txt"), "w") as f:
                    f.write(json.dumps(review, indent=4))
                    vol.commit()
            except Exception as e:
                print(f"Failed to perform review: {e}")
                return False

        ## IMPROVE WRITEUP
        if writeup == "latex" and improvement:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"*Starting Improvement*")
            try:
                perform_improvement(review, coder)
                generate_latex(
                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                    f.write(json.dumps(review))
                    vol.commit()
            except Exception as e:
                print(f"Failed to perform improvement: {e}")
                return False
        return True
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


def do_research_ideas(novel_ideas, base_dir, results_dir, client_model):
    for idea in novel_ideas:
        print(f"Processing idea: {idea['Name']}")
        try:
            with app.run():
                success = do_idea.remote(
                    base_dir,
                    results_dir,
                    idea,
                    "gpt-4o-2024-08-06",
                    client_model,
                    "latex",
                    True,
                )
                print(f"Completed idea: {idea['Name']}, Success: {success}")
        except Exception as e:
            print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")


def writeup(experiment_name, is_new_ideas=False, is_do_ideas=False):
    client_model = "gpt-4o-2024-08-06"

    base_dir = osp.join("templates", experiment_name)
    results_dir = osp.join("results", experiment_name)

    if is_new_ideas:
        with app.run():
            ideas = get_research_ideas.remote(
                base_dir, client_model, check_novelty=True
            )

        with open(osp.join(base_dir, "ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)

    if is_do_ideas:
        with open(osp.join(base_dir, "ideas.json"), "r") as f:
            ideas = json.loads(f.read())
        novel_ideas = [idea for idea in ideas]
        do_research_ideas(novel_ideas, base_dir, results_dir, client_model)


# Run baseline first before writeup
@app.function(
    gpu=modal.gpu.A10G(), image=dockerfile_image, volumes={"/data": vol}, timeout=3600
)
def do_experiment(command, experiment_name):
    try:
        working_directory = f"templates/{experiment_name}"
        result_1 = subprocess.run(
            command, cwd=working_directory, capture_output=True, text=True, check=True
        )

        vol.commit()
        print("STDOUT exp:", result_1.stdout)
        print("STDERR exp:", result_1.stderr)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code (exp):", e.returncode)
        print("Error output (exp):", e.stderr)


@app.function(
    gpu=modal.gpu.A10G(), image=dockerfile_image, volumes={"/data": vol}, timeout=3600
)
def do_plot(experiment_name):
    try:
        working_directory = f"templates/{experiment_name}"

        result_1 = subprocess.run(
            ["python", "plot.py"],
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=True,
        )

        vol.commit()
        print("STDOUT plt:", result_1.stdout)
        print("STDERR plt:", result_1.stderr)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code (plt):", e.returncode)
        print("Error output (plt):", e.stderr)


def run_baseline(experiment_name):
    base_dir = osp.join("templates", experiment_name)
    out_dir = f"/data/{base_dir}/run_0"

    command = ["python", "experiment.py", "--out_dir", out_dir]
    with app.run():
        do_experiment.remote(command, experiment_name)
        do_plot.remote(experiment_name)


if __name__ == "__main__":
    experiment_name = "seir"
    # run_baseline(experiment_name)
    writeup(experiment_name)

    # with app.run():
    #     create_img.remote()
