from re import L
from sklearn.metrics import accuracy_score, roc_auc_score
from subgraph_relevance import subgraph_mp_transcription
import numpy as np
import torch
from rdkit.Chem import AllChem
import networkx as nx
from rdkit import Chem
from matplotlib import pyplot as plt

def get_feat_order_local_best_guess(nn, g, alpha, H, transforms, mode='extr'):

    fo = []
    nb_nodes = g.nbnodes
    all_feats = np.arange(nb_nodes)
    if mode == 'prun':
        graph_score = subgraph_mp_transcription(nn, g, all_feats, alpha, H=H, transforms=transforms)

    while len(fo) < g.nbnodes:
        feat_list = list(frozenset(all_feats).difference(frozenset(fo)))
        max_score = -float("inf")
        max_feat = None
        for feat in feat_list:
            
            if mode == 'extr':
                S = fo + [feat]
                tmp_score = subgraph_mp_transcription(nn, g, S, alpha, H=H, transforms=transforms)
            else:
                S = list(frozenset(all_feats).difference(frozenset(fo + [feat])))
                tmp_score = -np.abs(subgraph_mp_transcription(nn, g, S, alpha, H=H, transforms=transforms) - graph_score)
            
            if tmp_score > max_score:
                max_score = tmp_score
                max_feat = feat
        fo += [max_feat]
    
    best_fo = torch.full( (nb_nodes,), -1)
    for i, fs in enumerate(fo):
        best_fo[fs] = nb_nodes - i
        
    return best_fo

def get_auac_aupc(nn, g, fo, task='extr', use_softmax=True):
    fo_ = fo.argsort(descending=True).numpy()
    if use_softmax: p = torch.nn.Softmax(dim=-1)(nn.forward(g.get_adj(), g.node_features)).data[g.label]
    else: p = nn.forward(g.get_adj(), g.node_features).data[g.label]

    if task == 'extr':
        # AUAC
        acs = []
        for i in range(1, len(fo_)+1):
            ind = fo_[:i]
            # print(len(ind))
            if use_softmax: p1 = torch.nn.Softmax(dim=-1)(nn.forward(g.get_adj()[ind][:,ind], g.node_features[ind])).data[g.label]
            else: p1 = nn.forward(g.get_adj()[ind][:,ind], g.node_features[ind]).data[g.label]
            acs.append(p1)
        auac = np.mean(acs)
        return auac, acs

    else:
        # AUPC
        nodeset = set(np.arange(len(g.get_adj())))
        pcs = []
        for i in range(0, len(fo_)):
            ind = list(nodeset - set(fo_[:i]))
            # print(len(ind))
            if use_softmax: p1 = torch.nn.Softmax(dim=-1)(nn.forward(g.get_adj()[ind][:,ind], g.node_features[ind])).data[g.label]
            else: p1 = nn.forward(g.get_adj()[ind][:,ind], g.node_features[ind]).data[g.label]
            pcs.append(np.abs(p - p1))
        aupc = np.mean(pcs)

        return aupc, pcs

def create_ground_truth(g):
    gr_tr = torch.zeros(25)
    gr_tr[20:] = 1
    all_feats = range(25)
    return gr_tr, all_feats

def topk(mat,k):
    if len(mat.shape) == 2:
        topk_values, linear_indices = mat.flatten().topk(k)
        indices = linear_indices.numpy() // mat.shape[-1], linear_indices.numpy() % mat.shape[-1]
        indices = [ (i,j) for i,j in zip(*indices)]
    else:
        topk_values, indices = torch.topk(mat, k)
        indices = indices.numpy()
        
    topk_values = topk_values.numpy()
    return topk_values, indices

def get_stats(gt, fo, all_feats):
    # make ids
    inds = torch.zeros(fo.shape)
    _, topkidx = topk(fo, len(gt.nonzero()))
    for idx in topkidx: inds[idx] = 1.

    acc = accuracy_score([gt[feat] for feat in all_feats],
                         [inds[feat] for feat in all_feats])
    auc = roc_auc_score([gt[feat] for feat in all_feats],
                        [fo[feat] for feat in all_feats])

    return acc, auc

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def get_all_walks(L, lamb, end_id=None, node_id=None, self_loops=True):
    """
    :param node_id: The id of the node we start with.
    :param L: The length of every walk.
    :param lamb: The adjacency matrix of the graph we consider.
    """
    if L == 1:
        return [[idx, idx] for idx in range(lamb.shape[-1])]

    def get_seq_of_nodes(tree):
        node_seq = [tree.id]
        while tree.parent is not None:
            tree = tree.parent
            node_seq.append(tree.id)
        return node_seq[::-1]
    
    def get_neighbors(id, lamb):
        x = torch.zeros(lamb.shape[1],1)
        x[id,0] = 1.
        neighbors = lamb.mm(x).nonzero()[:,0]
        return [int(id) for id in neighbors]

    if node_id is None:
        # Multiple start nodes
        num_of_nodes = lamb.shape[1]
        current_nodes = [None]*num_of_nodes
        for i in range(num_of_nodes): current_nodes[i] = Tree(); current_nodes[i].id = i
    else:
        # Starting in one node
        root = Tree()
        root.id = node_id
        current_nodes = [root]

    for l in range(L-1):
        leaf_nodes = []
        for node in current_nodes:
            for neighbor in get_neighbors(node.id, lamb):
                new_node = Tree()
                new_node.id = neighbor
                node.add_child(new_node)
                leaf_nodes.append(new_node)

        current_nodes = leaf_nodes

    all_walks = []
    for node in leaf_nodes:
        if end_id is None:
            all_walks.append(get_seq_of_nodes(node))
        else:
            if node.id == end_id:
                all_walks.append(get_seq_of_nodes(node))

    # filter out walks that include self loops
    if not self_loops:
        all_walks_filtered = []
        for w in all_walks:
            if len(set(w)) == len(w):
                all_walks_filtered.append(w)
        return all_walks_filtered

    return all_walks

def shrink(rx, ry, factor=11):
    """This function is used to make the walks smooth."""

    rx = np.array(rx)
    ry = np.array(ry)

    rx = 0.75 * rx + 0.25 * rx.mean()
    ry = 0.75 * ry + 0.25 * ry.mean()

    last_node = rx.shape[0] - 1
    concat_list_x = [np.linspace(rx[0], rx[0], 5)]
    concat_list_y = [np.linspace(ry[0], ry[0], 5)]
    for j in range(last_node):
        concat_list_x.append(np.linspace(rx[j], rx[j + 1], 5))
        concat_list_y.append(np.linspace(ry[j], ry[j + 1], 5))
    concat_list_x.append(np.linspace(rx[last_node], rx[last_node], 5))
    concat_list_y.append(np.linspace(ry[last_node], ry[last_node], 5))

    rx = np.concatenate(concat_list_x)
    ry = np.concatenate(concat_list_y)

    filt = np.exp(-np.linspace(-2, 2, factor) ** 2)
    filt = filt / filt.sum()

    rx = np.convolve(rx, filt, mode='valid')
    ry = np.convolve(ry, filt, mode='valid')

    return rx, ry

def plot_mutagenicity(g, relevances=None, width=12, shrinking_factor=11, factor=1, color=None, figname=None, dataset='Mutagenicity'):
    """plot the molecular, optional with relevances

    Args:
        relevances (List, optional): [[walk], relevance], like [[1,2,3],0.3] means the walk 1,2,3 has relevance 0.3. Defaults to None.
        width (int, optional): figure width. Defaults to 12.
        shrinking_factor (int, optional): shrink factor used for smoothing walk. Defaults to 11.
        factor (int, optional): multiply with the relevance score to make plot prettier. Defaults to 1.

    Returns:
        [type]: [description]
    """   
    node_label_dict = \
        {0:'C',1:'O',2:'Cl',3:'H',4:'N',5:'F',6:'Br',7:'S',8:'P',9:'I',10:'Na',11:'K',12:'Li',13:'Ca'} if dataset == 'Mutagenicity' else \
        {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}

    atoms = [node_label_dict[i] for i in g.node_tags]
    molecule = Chem.RWMol()
    for atom in atoms:
        molecule.AddAtom(Chem.Atom(atom))
    A = g.get_adj().nonzero()

    for x, y in A:
        if x < y:
            molecule.AddBond(int(x), int(y), Chem.rdchem.BondType.SINGLE)

    AllChem.Compute2DCoords(molecule)
    # compute 2D positions
    pos = []
    n_nodes = molecule.GetNumAtoms()
    for i in range(n_nodes):
        conformer_pos = molecule.GetConformer().GetAtomPosition(i)
        pos.append([conformer_pos.x, conformer_pos.y])
        
    pos = np.array(pos)

    # plotting
    fig_width = width
    pos_size = pos.max(axis=0) - pos.min(axis=0)
    fig_height = (width / pos_size[0]) * pos_size[1]
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(1, 1, 1)

    ####################################################################################################################
    # Utils
    ####################################################################################################################

    def _iterate_over_all_walks(ax, relevances, color=None):

        # visualization settings
        selfloopwidth = 0.25
        linewidth = 13.
        # start iteration over walks
        for walk_id, (walk, relevance) in enumerate(relevances):
            # get walk color
            if color is None:
                color = 'b' if relevance < 0 else 'r'
            # get opacity
            alpha = abs(relevance * factor)
            if alpha >1:
                alpha = 1.

            # split position vector in x and y part
            rx = np.array([pos[node][0] for node in walk])
            ry = np.array([pos[node][1] for node in walk])
            # plot g loops
            for i in range(len(rx) - 1):
                if rx[i] == rx[i + 1] and ry[i] == ry[i + 1]:
                    rx_tmp = rx[i] + selfloopwidth * np.cos(np.linspace(0, 2 * np.pi, 128))
                    ry_tmp = ry[i] + selfloopwidth * np.sin(np.linspace(0, 2 * np.pi, 128))
                    ax.plot(rx_tmp, ry_tmp, color=color, alpha=alpha, lw=linewidth, zorder=1.)
            # plot walks
            rx, ry = shrink(rx, ry, shrinking_factor)
            ax.plot(rx, ry, color=color, alpha=alpha, lw=linewidth, zorder=1.)
        return ax

    ####################################################################################################################
    # Main function code
    ####################################################################################################################

    # plot walks
    if relevances is not None:
        ax = _iterate_over_all_walks(ax, relevances, color)
    # ax = _iterate_over_all_walks(ax, [([0, 1, 23, 25], 0.09783325320052568)])

    G = nx.from_numpy_matrix(g.get_adj().numpy().astype(int)-np.eye(g.get_adj().shape[0]))
    # plot atoms
    collection = nx.draw_networkx_nodes(G, pos, node_color="w", alpha=0, node_size=500)
    collection.set_zorder(2.)
    # plot bonds
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color="w",
        width=4,
        style="dotted",
        node_size=300
    )
    # plot atom types
    pos_labels = pos - np.array([0.02, 0.05])
    nx.draw_networkx_labels(G, pos_labels, {i: name for i, name in enumerate(atoms)}, font_size=30)

    plt.axis('off')
    # plt.show()

    if figname is not None:
        plt.savefig(figname, dpi=600, format='eps',bbox_inches='tight')