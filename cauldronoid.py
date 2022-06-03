from pandas import DataFrame, concat
# Optional imports, so I can run this on my computer where I don't have pytorch
try:
    import torch as t
    from torch_geometric.data import Data
    has_torch = True
except ImportError:
    has_torch = False

def add_default_props(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        for prop_name, prop_value in atom.GetPropsAsDict(\
				includePrivate = False, includeComputed = False).items():
            atom.ClearProp(prop_name)
        atom.SetProp("atom_id", str(i))
    for i, bond in enumerate(mol.GetBonds()):
        for prop_name, prop_value in bond.GetPropsAsDict(\
				includePrivate = False, includeComputed = False).items():
            bond.ClearProp(prop_name)
        bond.SetProp("bond_id", str(i))
        bond.SetProp("start_atom", str(bond.GetBeginAtomIdx()))
        bond.SetProp("end_atom", str(bond.GetEndAtomIdx()))

def bind_rows_map(table_fun, name_fun, id_colname, xs):
	'''Apply a function to many objects, generating a table from each, and join
	these into a single table with a new column identifying the object. Inspired by
	Haskell's concatMap and dplyr's bind_rows.
	
	table_fun: Function mapping objects to tables (iterable of Pandas dataframes)
	name_fun: Function mapping objects to names (iterable of strings)
	id_colname: Name of new column to add containing the name of the object that the row comes from (string)
	xs: Objects to generate tables from
	
	Returns a Pandas dataframe'''
	names = map(name_fun, xs)
	tables = map(table_fun, xs)
	return concat(tables, axis = 0, keys = names).\
			reset_index(level = 0).\
			rename(columns = {"level_0": id_colname})
	
def molecule_table(rdkit_mol):
    return DataFrame([rdkit_mol.GetPropsAsDict(\
			includePrivate = False, includeComputed = False)])

def atom_table(rdkit_mol):
    return DataFrame(atom.GetPropsAsDict(includePrivate = False, includeComputed = False) \
                     for atom in rdkit_mol.GetAtoms())

def bond_table(rdkit_mol):
    return DataFrame(bond.GetPropsAsDict(includePrivate = False, includeComputed = False) \
                     for bond in rdkit_mol.GetBonds())

def tables2data(mol_id, molecule_table = None, atom_table = None,
                bond_table_oneway = None):
    '''From a molecule with n atoms, b bonds, j possible bond types and
    k possible elements, return
    an n by k one-hot matrix of element identites, a 2 by b matrix of bonds
    (represented by start and end indices), and a b by 4 matrix of one-hot
    bond types, wrapped in a Pytorch Geometric Data object, with molecule id
    and y value too'''
    if not has_torch:
        raise ImportError
    output = Data()
    output.mol_id = mol_id
    if molecule_table is not None:
        molecule_properties = molecule_table.drop(columns = "mol_id")
        # Only allows a single y value for now
        molecule_property_value = molecule_properties.to_numpy().reshape(())
        output.y = t.tensor(molecule_property_value, dtype = t.float32)
    # n by k matrix of one-hot element identities
    if atom_table is not None:
        # Sort the atom table so that position indicates index
        atom_table_sorted = atom_table.sort_values("atom_id")
        atom_properties = \
                atom_table_sorted.drop(columns = ["mol_id", "atom_id"])
        output.x = t.tensor(atom_properties.to_numpy(),
                          dtype = t.float32)
    if bond_table_oneway is not None:
        bond_table_reversed = bond_table_oneway.rename(\
                columns = {"start_atom": "end_atom", "end_atom": "start_atom"})
        bond_table_twoway = concat([bond_table_oneway, bond_table_reversed])
        edges = bond_table_twoway[["start_atom", "end_atom"]]
        # 2 by 2*b matrix containing the adjacency list
        output.edge_index = t.tensor(edges.to_numpy().transpose(),
                                     dtype = t.long)
        bond_properties = \
                bond_table_twoway.drop(columns = \
                        ["mol_id", "bond_id", "start_atom", "end_atom"])
        if bond_properties.shape[1] != 0:
            # 2*b by j matrix of one-hot bond types
            output.edge_attr = t.tensor(bond_properties.to_numpy(), dtype = t.float32)
    return output
