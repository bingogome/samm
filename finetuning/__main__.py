from argparse import ArgumentParser

from .dataset import add_export_args, export_dataset
from .eval_medsam2 import add_eval_args, eval_finetuned
from .example_data import add_example_args, setup_example_data
from .report import add_report_args, report
from .train_medsam2 import add_train_args, train_medsam2


def main():
    parser = ArgumentParser(description="SAMM finetuning tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_export_args(subparsers.add_parser("export-dataset", help="Export arrays to MedSAM2 NPZ training data"))
    add_example_args(subparsers.add_parser("setup-example-data", help="Download and export MSD example data"))
    add_train_args(subparsers.add_parser("train-medsam2", help="Launch MedSAM2 finetuning"))
    add_eval_args(subparsers.add_parser("eval-finetuned", help="Evaluate a MedSAM2 checkpoint with 2D prompts"))
    add_report_args(subparsers.add_parser("report", help="Summarize a finetuning run"))
    args = parser.parse_args()
    {
        "export-dataset": export_dataset,
        "setup-example-data": setup_example_data,
        "train-medsam2": train_medsam2,
        "eval-finetuned": eval_finetuned,
        "report": report,
    }[args.command](args)


if __name__ == "__main__":
    main()
