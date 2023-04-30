import os.path
from pandas import DataFrame, concat, read_csv
# Optional imports, so I can run this on my computer where I don't have pytorch
try:
    from rdkit.Chem.rdmolops import SanitizeMol, RemoveHs
    from rdkit.Chem.rdchem import BondType, Atom, Mol, EditableMol, Conformer
    from rdkit.Geometry.rdGeometry import Point3D
    has_rdkit = True
except ImportError:
    has_rdkit = False
try:
    import torch as t
    from torch_geometric.data import Data
    has_torch = True
except ImportError:
    has_torch = False
try:
    import ase
    has_ase = True
except ImportError:
    has_ase = False

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
        "end_atom",
        "formal_charge",
        "n_hydrogen",
        "bond_type"])
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
    def __init__(self, molecule_table = None, one_atom_table = None,
                 two_atom_table = None, name = None, special_colnames = None):
        self.special_colnames = dict((i, i) for i in default_names)
        if special_colnames is not None:
            # Add or replace special column names according to the special_colnames argument
            self.special_colnames.update(special_colnames)
            # I want there to be a way to remove special column names, so if
            # they're given as None, remove them
            to_remove = [key \
                    for key, value in self.special_colnames.items() \
                    if value is None]
            for key in to_remove:
                del self.special_colnames[key]
        if molecule_table is not None:
            # This represents ONE molecule!
            assert molecule_table.shape[0] == 1
            self.molecule_table = molecule_table
        else:
            self.molecule_table = None
        # Get the name from the name argument, or, if not available, from the molecule table
        if name is not None:
            self.name = str(name)
        elif molecule_table is not None and \
                self.special_colnames["mol_id"] in molecule_table.columns:
            # mol_id column: pandas Series object
            mol_id_col = molecule_table[self.special_colnames["mol_id"]]
            # Retrieve the first value
            self.name = str(mol_id_col.item())
        else:
            self.name = None
        self.one_atom_table = one_atom_table
        # TO DO: If they're bonded check if it's a connected component?
        self.two_atom_table = two_atom_table
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
                [[self.special_colnames["x"], self.special_colnames["y"],
                    self.special_colnames["z"]]].\
                to_numpy()
    def get_element_symbols(self):
        return self.one_atom_table[self.special_colnames["symbol"]].to_list()
    def get_atomic_numbers(self):
        return [symbols2numbers[symbol] for symbol in self.get_element_symbols()]
    def get_rdkit(self, sanitize = True, removeHs = True):
        '''Return an rdkit molecule'''

        if not has_rdkit:
            raise ImportError("RDKit not installed")

        mol_reconstructed_editable = EditableMol(Mol())

        atom_id_col = self.special_colnames["atom_id"]
        symbol_col = self.special_colnames["symbol"]
        formal_charge_col = self.special_colnames["formal_charge"]
        # Indices of atoms will be needed to add bonds
        atom_indices = dict()
        # Types of atom table columns, excluding special columns like positions
        one_tbl_column_types = non_special_column_types(self.get_one_atom_table(),
                                                        self.special_colnames.keys())
        # Using itertuples instead of iterrows so the types are consistent in each row
        for i, rowtuple in enumerate(self.get_one_atom_table().itertuples()):
            # Converting to a dictionary so I can access values by with square brackets
            row = rowtuple._asdict()
            # Need type conversion because I don't know how to control pandas
            # types and it may infer a floating point type for the column
            element = str(row[symbol_col])
            atom_id = str(row[atom_id_col])
            formal_charge = int(row[formal_charge_col])
            new_rdkit_atom = Atom(element)
            # If removeHs is off, be completely literal: hydrogens present if
            # and only if specified. Motivated by the chelating oxygens in
            # CSD's acac ligands, which RDKit incorrectly adds oxygens to.
            # Possibly this should be a separate option but for now, this seems
            # like a good compromise between preventing errors, and enabling
            # working with molecules in the organic chemist's H-free data
            # structure
            if not removeHs:
                new_rdkit_atom.SetNoImplicit(True)
            new_rdkit_atom.SetFormalCharge(formal_charge)
            new_rdkit_atom.SetProp("_Name", atom_id)
            # Set non-special atom properties
            for colname, dtype in one_tbl_column_types.items():
                set_rdkit_prop(new_rdkit_atom, dtype, colname, row[colname])

            atom_indices[atom_id] = i
            mol_reconstructed_editable.AddAtom(new_rdkit_atom)

        bond_id_col = self.special_colnames["bond_id"]
        start_atom_col = self.special_colnames["start_atom"]
        end_atom_col = self.special_colnames["end_atom"]
        bond_order_col = self.special_colnames["bond_type"]
        # Using itertuples instead of iterrows so the types are consistent in each row
        for rowtuple in self.get_two_atom_table().itertuples():
            # Converting to a dictionary so I can access values by with square brackets
            row = rowtuple._asdict()
            start_atom = str(row[start_atom_col])
            end_atom = str(row[end_atom_col])
            order_string = str(row[bond_order_col])
            order_rdkit = BondType.names[order_string]
            # Relies on the atoms containing their index in the name
            start_atom_index = atom_indices[start_atom]
            end_atom_index = atom_indices[end_atom]
            mol_reconstructed_editable.AddBond(start_atom_index,
                    end_atom_index, order_rdkit)
        mol_reconstructed = mol_reconstructed_editable.GetMol()

        # Check column names of one atom table for x, y and z to decide whether
        # to add a conformation
        x_col = self.special_colnames["x"]
        y_col = self.special_colnames["y"]
        z_col = self.special_colnames["z"]
        if x_col in self.get_one_atom_table().columns and \
                y_col in self.get_one_atom_table().columns and \
                z_col in self.get_one_atom_table().columns:
            # Create the conformer
            n_atoms = mol_reconstructed.GetNumAtoms()
            conf = Conformer(n_atoms)
            conf.Set3D(True)
            conf.SetId(0)

            # Second loop over atom table to add positions to the conformer
            for i, rowtuple in enumerate(self.get_one_atom_table().itertuples()):
                # Converting to a dictionary so I can access values by with square brackets
                row = rowtuple._asdict()
                conf.SetAtomPosition(i,
                        Point3D(row[x_col], row[y_col], row[z_col]))

            # Add the conformer
            mol_reconstructed.AddConformer(conf)

        # Types of bond table columns, excluding special columns like positions
        two_tbl_column_types = non_special_column_types(self.get_two_atom_table(),
                                                        self.special_colnames.keys())
        for rowtuple in self.get_two_atom_table().itertuples():
            # Converting to a dictionary so I can access values by with square brackets
            row = rowtuple._asdict()
            bond_id = str(row[bond_id_col])
            start_atom = str(row[start_atom_col])
            end_atom = str(row[end_atom_col])
            # Relies on the atoms containing their index in the name
            start_atom_index = atom_indices[start_atom]
            end_atom_index = atom_indices[end_atom]
            bond = mol_reconstructed.GetBondBetweenAtoms(start_atom_index, end_atom_index)
            if bond is None:
                bond = mol_reconstructed.GetBondBetweenAtoms(end_atom_index, start_atom_index)
            # Set non-special bond properties
            for colname, dtype in two_tbl_column_types.items():
                set_rdkit_prop(bond, dtype, colname, row[colname])

            bond.SetProp("_Name", bond_id)

        # Add all molecule properties
        mol_reconstructed.SetProp("_Name", self.get_name())
        # Retrieve the type of each column of the table
        mol_tbl_column_types = non_special_column_types(self.get_molecule_table(),
                                                        self.special_colnames.keys())

        # Looping over rows of the molecule table, but there should only be one
        # row
        for rowtuple in self.get_molecule_table().itertuples(index = False):
            row = rowtuple._asdict()
            for colname, dtype in mol_tbl_column_types.items():
                set_rdkit_prop(mol_reconstructed, dtype, colname, row[colname])

        # Maybe sanitize
        if sanitize:
            SanitizeMol(mol_reconstructed)

        # Maybe remove hydrogens
        if removeHs:
            outmol = RemoveHs(mol_reconstructed)
        else:
            outmol = mol_reconstructed

        return outmol
    def get_torchgeom(self):
        '''Return a Pytorch geometric Data object'''
        return tables2data(self.name, self.molecule_table, self.one_atom_table,
                self.two_atom_table)
    def get_ase(self):
        '''Returns an Atomic Simulation Environment Atoms object'''
        if not has_ase:
            raise ImportError("Atomic simulation environment Python library required to return ase objects")
        return ase.Atoms(self.get_element_symbols(), self.get_positions())
    def get_pymatgen(self):
        '''Returns a Pymatgen Molecule object'''
        pass

def add_positions(rdkit_mol, conf_id = 0):
    for pos_vec, atom in \
            zip(list(rdkit_mol.GetConformer(conf_id).GetPositions()), rdkit_mol.GetAtoms()):
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
    if rdkit_mol.HasProp("_Name") and rdkit_mol.GetProp("_Name") != "":
        name = rdkit_mol.GetProp("_Name")
    else:
        name = None
    cauldronoid_mol = Molecule(molecule_table(rdkit_mol),
            atom_table(rdkit_mol), bond_table(rdkit_mol), name)
    return cauldronoid_mol

def add_default_props(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("atom_id", str(i))
        atom.SetProp("formal_charge", str(atom.GetFormalCharge()))
        atom.SetProp("n_hydrogen", str(atom.GetTotalNumHs()))
    for i, bond in enumerate(mol.GetBonds()):
        bond.SetProp("bond_id", str(i))
        bond.SetProp("start_atom", str(bond.GetBeginAtomIdx()))
        bond.SetProp("end_atom", str(bond.GetEndAtomIdx()))
        bond.SetProp("bond_type", bond.GetBondType().name)

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
        # Output a 2D vector containing the molecule values
        molecule_property_value = molecule_properties.to_numpy()
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

def tables2mols(molecule_table, one_atom_table,
        two_atom_table, special_colnames = None):
    if special_colnames is None:
        mol_id_colname = "mol_id"
    else:
        mol_id_colname = special_colnames["mol_id"]
    mol_ids = set(molecule_table[mol_id_colname])
    # Without "iter", I get:
    # TypeError: 'str' object is not callable
    # I'm not sure where this error happens
    molecule_dict = dict(iter(molecule_table.groupby(mol_id_colname)))
    one_atom_dict = dict(iter(one_atom_table.groupby(mol_id_colname)))
    two_atom_dict = dict(iter(two_atom_table.groupby(mol_id_colname)))
    return [Molecule(\
        molecule_dict[mol_id],
        one_atom_dict[mol_id],
        two_atom_dict[mol_id]) \
                for mol_id in mol_ids]

def mols2tables(mols):
    # Get the molecule colname from the first molecule, assuming it's the same for all
    mol_id_colname = mols[0].special_colnames["mol_id"]
    assert all(mol.special_colnames["mol_id"] == mol_id_colname for mol in mols)
    molecule_table = bind_rows_map(Molecule.get_molecule_table,
            Molecule.get_name, mol_id_colname, mols)
    one_atom_table = bind_rows_map(Molecule.get_one_atom_table,
            Molecule.get_name, mol_id_colname, mols)
    two_atom_table = bind_rows_map(Molecule.get_two_atom_table,
            Molecule.get_name, mol_id_colname, mols)
    return molecule_table, one_atom_table, two_atom_table

def files2mols(directory = None, prefix = None, suffixes = ("_mol_tbl.csv.gz",
    "_one_tbl.csv.gz", "_two_tbl.csv.gz"), molecule_table_path = None,
    one_atom_table_path = None, two_atom_table_path = None):
    if prefix is not None:
        if directory is None:
            # os.path.join won't add slashes if you give an empty string so this is safe
            directory = ""
        molecule_table_path = os.path.join(directory, prefix + suffixes[0])
        one_atom_table_path = os.path.join(directory, prefix + suffixes[1])
        two_atom_table_path = os.path.join(directory, prefix + suffixes[2])
    molecule_table = read_csv(molecule_table_path)
    one_atom_table = read_csv(one_atom_table_path)
    two_atom_table = read_csv(two_atom_table_path)
    return tables2mols(molecule_table, one_atom_table, two_atom_table)

def mols2files(mols, prefix, directory = None, suffixes = ("_mol_tbl.csv.gz",
    "_one_tbl.csv.gz", "_two_tbl.csv.gz"), molecule_table_path = None,
    one_atom_table_path = None, two_atom_table_path = None):
    molecule_table, one_atom_table, two_atom_table = mols2tables(mols)

    if directory is None:
        # os.path.join won't add slashes if you give an empty string so this is safe
        directory = ""
    molecule_table_path = os.path.join(directory, prefix + suffixes[0])
    one_atom_table_path = os.path.join(directory, prefix + suffixes[1])
    two_atom_table_path = os.path.join(directory, prefix + suffixes[2])

    molecule_table.to_csv(molecule_table_path, index = False)
    one_atom_table.to_csv(one_atom_table_path, index = False)
    two_atom_table.to_csv(two_atom_table_path, index = False)

def set_rdkit_prop(obj, dtype, name, value):
    '''Set the value of a property of an RDKit molecule, atom or bond

    obj: The molecule, atom or bond (anything with SetProp etc methods)

    dtype: A numpy datatype

    name: The name of the property

    value: The value to set the property to

    Modifies in place, no return value'''
    if dtype.kind == "b":
        obj.SetBoolProp(name, value)
    elif dtype.kind == "i":
        obj.SetIntProp(name, value)
    elif dtype.kind == "u":
        obj.SetUnsignedProp(name, value)
    elif dtype.kind == "f":
        obj.SetDoubleProp(name, value)
    elif dtype.kind == "S" or dtype.kind == "U":
        obj.SetProp(name, value)
    elif dtype.kind == "O":
        # Column of type object not guaranteed to be strings, but can be converted
        # Would prefer if this column type were not used, but don't know how to prevent it
        obj.SetProp(name, str(value))
    else:
        raise ValueError(f"dtype.kind {dtype.kind} not recognized")

def non_special_column_types(tbl, excluded_colnames):
    return dict((colname, dtype) \
                for colname, dtype in tbl.dtypes.to_dict().items() \
                if colname not in excluded_colnames)

