import sys
import pandas as pd
from rdkit.Chem.rdmolfiles import SDMolSupplier

# Horrible, split into multiple lines
table = pd.DataFrame(dict([(sys.argv[2], mol.GetProp("_Name"))] + list(mol.GetPropsAsDict().items())) for mol in SDMolSupplier(sys.argv[1]))
table.to_csv(sys.argv[3], index = False)
