from model.dgl.DGLModel import DGLModel


class MGCN(DGLModel):
    def __init__(
            self,
            n_tasks: int,
            feats: int,
            n_layers: int = 3,
            classifier_hidden_feats: int = 64,
            num_node_types: int = 100,
            num_edge_types: int = 3000,
            cutoff: float = 5.0,
            gap: float = 1.0,
            predictor_hidden_feats: int = 64,
            **kwargs
    ):
        from dgllife.model import MGCNPredictor
        super().__init__(
                model=MGCNPredictor(
                        feats=feats,
                        n_layers=n_layers,
                        classifier_hidden_feats=classifier_hidden_feats,
                        n_tasks=n_tasks,
                        num_node_types=num_node_types,
                        num_edge_types=num_edge_types,
                        cutoff=cutoff,
                        gap=gap,
                        predictor_hidden_feats=predictor_hidden_feats
                ),
                **kwargs
        )

    def _predict(self, graphs):
        node_types = graphs.ndata.pop('node_type').to(self.device)
        edge_distances = graphs.edata.pop('distance').to(self.device)
        return self.model(graphs, node_types, edge_distances)


def alchemy_nodes(mol):
    """
    Featurization for all atoms in a molecule.
    The atom indices will be preserved.
    From dgllife.data.alchemy

    :param mol: RDKit molecule object
    :return: Dictionary for atom features
    """
    from collections import defaultdict
    import os.path as osp
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures, AllChem
    from rdkit import Chem
    import numpy as np
    from dgl import backend as F
    from dgllife.utils.featurizers import atom_type_one_hot, atom_hybridization_one_hot, atom_is_aromatic

    atom_feats_dict = defaultdict(list)
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    mol_feats = mol_featurizer.GetFeaturesForMol(mol)

    if AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100) == -1:
        print("[ERROR] Failed: Use distance geometry to obtain initial coordinates for a molecule")

    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1

    for i in range(len(mol_feats)):
        if mol_feats[i].GetFamily() == 'Donor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor[u] = 1
        elif mol_feats[i].GetFamily() == 'Acceptor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_acceptor[u] = 1

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        atom_type = atom.GetAtomicNum()
        num_h = atom.GetTotalNumHs()
        atom_feats_dict['node_type'].append(atom_type)

        h_u = []
        h_u += atom_type_one_hot(atom, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl'])
        h_u.append(atom_type)
        h_u.append(is_acceptor[u])
        h_u.append(is_donor[u])
        h_u += atom_is_aromatic(atom)
        h_u += atom_hybridization_one_hot(atom, [Chem.rdchem.HybridizationType.SP,
                                                 Chem.rdchem.HybridizationType.SP2,
                                                 Chem.rdchem.HybridizationType.SP3])
        h_u.append(num_h)
        atom_feats_dict['n_feat'].append(F.tensor(np.array(h_u).astype(np.float32)))

    atom_feats_dict['n_feat'] = F.stack(atom_feats_dict['n_feat'], dim=0)
    atom_feats_dict['node_type'] = F.tensor(np.array(
            atom_feats_dict['node_type']).astype(np.int64))

    return atom_feats_dict


def alchemy_edges(mol, self_loop=False):
    """
    Featurization for all bonds in a molecule.
    The bond indices will be preserved.
    From dgllife.data.alchemy

    :param mol: RDKit molecule object
    :param self_loop: Whether to add self loops. Default to be False.
    :return: Dictionary for bond features
    """
    from collections import defaultdict
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    from dgl import backend as F

    bond_feats_dict = defaultdict(list)

    if AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100) == -1:
        print("[ERROR] Failed: Use distance geometry to obtain initial coordinates for a molecule")

    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    geom = mol_conformers[0].GetPositions()

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None:
                bond_type = None
            else:
                bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append([
                    float(bond_type == x)
                    for x in (Chem.rdchem.BondType.SINGLE,
                              Chem.rdchem.BondType.DOUBLE,
                              Chem.rdchem.BondType.TRIPLE,
                              Chem.rdchem.BondType.AROMATIC, None)
            ])
            bond_feats_dict['distance'].append(
                    np.linalg.norm(geom[u] - geom[v]))

    bond_feats_dict['e_feat'] = F.tensor(
            np.array(bond_feats_dict['e_feat']).astype(np.float32))
    bond_feats_dict['distance'] = F.tensor(
            np.array(bond_feats_dict['distance']).astype(np.float32)).reshape(-1, 1)

    return bond_feats_dict
