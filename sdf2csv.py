import sys
from argparse import ArgumentParser
import pandas as pd
from rdkit.Chem.rdmolfiles import ForwardSDMolSupplier

argument_parser = ArgumentParser(description = " ".join([\
        "Given molecules in SDF format, create a table of molecule properties in csv format.",
        "Each molecule will be one row in the output.",
        "The output CSV file includes a header row.",
        "Only includes molecule level properties, not atom or bond.",
        "Molecules are read from stdin and output to stdout.",
        "Does not include computed properties (names start with _ or __).",
        "Does not include molecule name unless a corresponding column name is given."]))

argument_parser.add_argument("--name", "-n", help = \
    "String to use as a column name in output for the molecule name (\"_Name\" attribute)")

argument_dict = vars(argument_parser.parse_args())
name_col = argument_dict["name"]

inmols = ForwardSDMolSupplier(sys.stdin.buffer)
table = pd.DataFrame(dict([(name_col, mol.GetProp("_Name"))] + list(mol.GetPropsAsDict().items())) for mol in inmols)
table.to_csv(sys.stdout, index = False)
