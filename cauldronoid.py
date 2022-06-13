from pandas import DataFrame, concat
# Optional imports, so I can run this on my computer where I don't have pytorch
try:
    import torch as t
    from torch_geometric.data import Data
    has_torch = True
except ImportError:
    has_torch = False
import ase

# Semantically special column names
default_names = frozenset([\
        "mol_id",
        "atom_id",
        "symbol",
        "x",
        "y",
        "z",
        "bond_id",
        "start_atom",
        "end_atom"])
# Periodic table
symbols2numbers = {\
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
        "Fr": 87,
        "Ra": 88,
        "Ac": 89,
        "Th": 90,
        "Pa": 91,
        "U": 92,
        "Np": 93,
        "Pu": 94,
        "Am": 95,
        "Cm": 96,
        "Bk": 97,
        "Cf": 98,
        "Es": 99,
        "Fm": 100,
        "Md": 101,
        "No": 102,
        "Lr": 103,
        "Rf": 104,
        "Db": 105,
        "Sg": 106,
        "Bh": 107,
        "Hs": 108,
        "Mt": 109,
        "Ds": 110,
        "Rg": 111,
        "Cn": 112,
        "Nh": 113,
        "Fl": 114,
        "Mc": 115,
        "Lv": 116,
        "Ts": 117,
        "Og": 118}

class Molecule:
    def __init__(self, name = None, molecule_table = None, one_atom_table = None,
                 two_atom_table = None, special_colnames = None):
        self.special_colnames = dict((i, i) for i in default_names)
        if special_colnames is not None:
            # Add or replace special column names according to the special_colnames argument
            self.special_colnames.update(special_colnames)
            # I want there to be a way to remove special column names, so if they're given as None, remove them
            to_remove = [key for key, value in self.special_colnames.items() if value is None]
            for key in to_remove:
                del x[key]
        if molecule_table is not None:
            # This represents ONE molecule!
            assert molecule_table.shape[0] == 1
            self.molecule_table = molecule_table
        else:
            self.molecule_table = None
        # Get the name from the name argument, or, if not available, from the molecule table
        if name is not None:
            self.name = name
        elif molecule_table is not None and self.special_colnames["mol_id"] in molecule_table.columns:
            self.name = molecule_table[self.special_colnames["mol_id"]][0]
        else:
            self.name = None
        self.one_atom_table = one_atom_table
        # TO DO: If they're bonded check if it's a connected component?
        self.two_atom_table = two_atom_table
    def get_name(self):
        return self.name
    # Retrieving the tables
    def get_molecule_table(self):
        return self.molecule_table
    def get_one_atom_table(self):
        return self.one_atom_table
    def get_two_atom_table(self):
        return self.two_atom_table
    # Standard properties
    def get_name(self):
        return self.name
    def get_positions(self):
        return self.one_atom_table\
                [[self.special_colnames["x"], self.special_colnames["y"], self.special_colnames["z"]]].\
                to_numpy()
    def get_element_symbols(self):
        return self.one_atom_table[self.special_colnames["symbol"]].to_list()
    def get_atomic_numbers(self):
        return [symbols2numbers[symbol] for symbol in self.get_element_symbols()]
    def get_rdkit(self):
        '''Return an rdkit molecule'''
        pass
    def get_torchgeom(self):
        '''Return a Pytorch geometric Data object'''
        return tables2data(self.name, self.molecule_table, self.one_atom_table, self.two_atom_table)
    def get_ase(self):
        '''Returns an Atomic Simulation Environment Atoms object'''
        return ase.Atoms(self.get_element_symbols(), self.get_positions())
    def get_pymatgen(self):
        '''Returns a Pymatgen Molecule object'''
        pass

def add_positions(rdkit_mol, conf_id = 0):
    for pos_vec, atom in zip(list(rdkit_mol.GetConformer(conf_id).GetPositions()), rdkit_mol.GetAtoms()):
        atom.SetDoubleProp("x", pos_vec[0])
        atom.SetDoubleProp("y", pos_vec[1])
        atom.SetDoubleProp("z", pos_vec[2])

def add_symbols(rdkit_mol):
    for atom in rdkit_mol.GetAtoms():
        atom.SetProp("symbol", atom.GetSymbol())

def rdkit2cauldronoid(rdkit_mol):
    add_default_props(rdkit_mol)
    if rdkit_mol.GetNumConformers() > 0:
        add_positions(rdkit_mol)
    add_symbols(rdkit_mol)
    cauldronoid_mol = Molecule(molecule_table = molecule_table(rdkit_mol), one_atom_table = atom_table(rdkit_mol), two_atom_table = bond_table(rdkit_mol)) 
    return cauldronoid_mol

def add_default_props(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("atom_id", str(i))
    for i, bond in enumerate(mol.GetBonds()):
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
        raise ImportError("Cannot convert to Pytorch Geometric data because it is not installed")
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
