import argparse
import json

from base_processing import BaseProcess


class Pipeline:
    def __init__(self, path):
        self.path = path
        self.process = BaseProcess()

    def run(self, filename: str = "output_data.json"):
        data = self.process.process_document(self.path)
        with open(filename) as file:
            expected = json.load(file)
            if data == expected:
                print("passed")
            else:
                self.write_markup(filename, data)
                print("markup updated")

    def shielding_markup(self, data):
        if isinstance(data, list):
            iterator = enumerate(data)
        else:
            iterator = data.items()
        for i, v in iterator:
            if isinstance(v, str):
                data[i] = v.replace("\\", "\\\\")
                data[i] = v.replace('"', r"\"")
            elif isinstance(v, (list, dict)):
                self.shielding_markup(v)

    def write_markup(self, filename, data, shielding=True):
        if shielding:
            self.shielding_markup(data)
        res = json.dumps(data, indent=4).encode("utf8").decode("unicode-escape")
        with open(filename, "w") as file:
            file.write(res)


if __name__ == "__main__":
    a_parser = argparse.ArgumentParser(description="Parse document")
    a_parser.add_argument("--invoice_path", "-p", help="Path to file", required=True)
    args = a_parser.parse_args()
    pipeline = Pipeline(args.invoice_path)
    pipeline.run()
