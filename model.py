import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.nn import LayerNorm
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx, unbatch_edge_index


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False, reflection_equiv=True):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.reflection_equiv = reflection_equiv
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # self.edge_coord_mlp = nn.Sequential(
        #     nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, hidden_nf),
        #     act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        self.cross_product_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        ) if not self.reflection_equiv else None

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out


    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch=None):
        row, col = edge_index
        tmp = self.coord_mlp(edge_feat)

        trans = coord_diff * tmp

        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = self.coord2cross(coord, edge_index, batch=batch)

            phi_cross = self.cross_product_mlp(edge_feat)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross)  # * self.coords_range

            trans = trans + coord_cross * phi_cross

        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)


        coord = coord + agg

        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            # norm = torch.sqrt(radial).detach() + self.epsilon
            norm = torch.sqrt(radial).detach() + 1
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def coord2cross(self, coord, edge_index, norm_constant=1, batch=None):

        mean = unsorted_segment_sum(coord, batch,
                                    num_segments=batch.max() + 1,
                                    aggregation_method='mean')
        row, col = edge_index
        cross = torch.cross(coord[row] - mean[batch[row]],
                            coord[col] - mean[batch[col]], dim=1)
        norm = torch.linalg.norm(cross, dim=1, keepdim=True)
        cross = cross / (norm + norm_constant)
        return cross

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, batch=batch)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 residual=True, attention=True, normalize=False, tanh=True, reflection_equiv=True):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh, reflection_equiv=reflection_equiv))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr, batch=None):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            # print(f"##### egnn layer {i} #####")
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, batch=batch)
        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments, aggregation_method=None):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm

    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, bn=False):
        '''
        3 layer MLP

        Args:
            input_dim: # input layer nodes
            hidden_dim: # hidden layer nodes
            output_dim: # output layer nodes
            activation: activation function
            layer_norm: bool; if True, apply LayerNorm to output
        '''

        # init superclass and hidden/ output layers
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

        self.bn = bn
        if self.bn:
            self.bn = nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.997)

        # init activation function reset parameters
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):

        # reset model parameters
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x):

        # forward prop x
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        x = self.lin3(x)

        return x


class EGNNPooling(torch.nn.Module):
    def __init__(self, device='cpu', hidden_dim=32, stride=2, kernel=3, padding=1, attn=False):
        super(EGNNPooling, self).__init__()

        self.hidden_dim = hidden_dim
        self.stride = stride
        self.kernel = kernel
        self.padding = padding
        self.device = device

        self.egnnse3 = EGNN(in_node_nf=hidden_dim, hidden_nf=hidden_dim, out_node_nf=hidden_dim, in_edge_nf=hidden_dim,
                         attention=attn, reflection_equiv=False)


        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
        self.edge_mlp_after_pooling = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)

        self.bn_edge = LayerNorm(in_channels=hidden_dim, mode="node")
        self.bn_h = LayerNorm(in_channels=hidden_dim, mode="node")
        self.bn_edge_after_pooling = LayerNorm(in_channels=hidden_dim, mode="node")

    def forward(self, h, coords, batch=None, batched_data=None, edge_index=None):

        ## get initial h and coords for pooling node

        # get number of node in one input graph and number of pooling node
        num_node = int(torch.div(h.shape[0], (batch[-1] + 1), rounding_mode='floor'))
        num_pool_node = int(torch.div(num_node + 2 * self.padding - self.kernel, self.stride, rounding_mode='floor')) + 1

        # build mapping matrix to map original node to initial pooling node
        M = torch.zeros((num_pool_node, num_node + 2 * self.padding)).double().to(self.device)
        for i in range(num_pool_node):
            M[i, i * self.stride:(i * self.stride + self.kernel)] = 1 / self.kernel
        # create index to get coords for one graph and padding node (padding mode: same)
        h = h.view((batch[-1] + 1), num_node, -1) # B x n x F
        coords = coords.view((batch[-1] + 1), num_node, -1)  # B x n x 3
        index = [0] * self.padding + list(range(0, num_node)) + [num_node - 1] * self.padding
        coords = coords[:, index, :]
        h = h[:, index, :]
        coords_pool = M @ coords # broadcast matrix multiplication
        h_pool = M @ h


        if edge_index is None:
            edge_index = from_networkx(nx.complete_graph(coords.shape[1])).edge_index.to(self.device)
            edge_index_unbatch = edge_index.unsqueeze(0).repeat((batch[-1] + 1), 1, 1)
        else:
            edge_index_unbatch = torch.stack(unbatch_edge_index(edge_index, batch), dim=0)

        edge_index_unbatch = edge_index_unbatch + self.padding 


        # connect pooling nodes to input graph nodes
        row, col = torch.where(M > 0)
        index_pool = torch.vstack((row + num_node + 2 * self.padding, col))
        index_pool = torch.cat((index_pool, torch.flip(index_pool, dims=[0])), dim=1)
        edge_index_unbatch = torch.cat((edge_index_unbatch, index_pool.unsqueeze(0).repeat(edge_index_unbatch.shape[0], 1, 1)), dim=2) # B x 2 x num_edges
        h = torch.cat((h, h_pool), dim=1) # B x (n + n_pool) x F
        coords = torch.cat((coords, coords_pool), dim=1) # B x (n + n_pool) x 3

        datalist = []
        for i in range(batch[-1] + 1):
            datalist.append(Data(edge_index=edge_index_unbatch[i]))


        # perform egnn
        data = Batch.from_data_list(datalist).to(self.device)
        h = h.view(-1, self.hidden_dim)
        coords = coords.view(-1, 3)
        row, col = data.edge_index
        out = torch.cat([h[row], h[col]], dim=1)
        edge_attr = self.edge_mlp(out)
        edge_attr = self.bn_edge(edge_attr)
        h, coords = self.egnnse3(h, coords, data.edge_index, edge_attr, data.batch)

        h = self.bn_h(h)


        # keep pooling node
        h = h.view(batch[-1] + 1, -1, h.shape[1])
        h_pool = h[:, (num_node + 2 * self.padding):, :].reshape(-1, h.shape[2])
        coords = coords.view(batch[-1] + 1, -1, coords.shape[1])
        coords_pool = coords[:, (num_node + 2 * self.padding):, :].reshape(-1, coords.shape[2])


        return h_pool, coords_pool

class EGNNUnPooling(torch.nn.Module):
    def __init__(self, device='cpu', hidden_dim=32, stride=2, kernel=3, padding=1, output_padding=1, attn=False):
        super(EGNNUnPooling, self).__init__()

        self.hidden_dim = hidden_dim
        self.stride = stride
        self.kernel = kernel
        self.padding = padding
        self.output_padding = output_padding
        self.device = device


        self.egnnse3 = EGNN(in_node_nf=hidden_dim, hidden_nf=hidden_dim, out_node_nf=hidden_dim, in_edge_nf=hidden_dim,
                         attention=attn, reflection_equiv=False)


        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
        self.edge_mlp_after_pooling = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)

        self.bn_edge = LayerNorm(in_channels=hidden_dim, mode="node")
        self.bn_h = LayerNorm(in_channels=hidden_dim, mode="node")
        self.bn_edge_after_pooling = LayerNorm(in_channels=hidden_dim, mode="node")

    def forward(self, h, coords, batch=None, batched_data=None, edge_index=None):

        # initialize coords for pooling node
        num_node = int(torch.div(h.shape[0], (batch[-1] + 1), rounding_mode='floor'))

        # size after padding
        aug_size = (num_node * self.stride - 1) + 2 * (self.kernel - self.padding - 1) + self.output_padding
        out_size = (num_node - 1) * self.stride - 2 * self.padding + (self.kernel - 1) + self.output_padding + 1
        M = torch.zeros((out_size, aug_size)).double().to(self.device)
        for i in range(out_size):
            M[i, i:(i + self.kernel)] = 1 / self.kernel

        h = h.view((batch[-1] + 1), num_node, -1) # B x n x F
        coords = coords.view((batch[-1] + 1), num_node, -1)  # B x n x 3

        ##### add same position and h on boundry, add average position and h in between #####
        avg = torch.stack([coords[:, 0:-1, :], coords[:, 1:, :]], dim=2).mean(dim=2) # B x (n-1) x 3
        tmp = torch.stack([coords[:, 0:-1, :], avg], dim=2) # B x (n-1) x 2 x 3
        tmp = torch.flatten(tmp, start_dim=1, end_dim=2) # B x 2*(n-1) x 3
        coords = torch.cat([coords[:, 0:1, :],
                            tmp,
                            coords[:, -1:, :].repeat(1,3,1)], dim=1)

        avg = torch.stack([h[:, 0:-1, :], h[:, 1:, :]], dim=2).mean(dim=2) # B x (n-1) x F
        tmp = torch.stack([h[:, 0:-1, :], avg], dim=2) # B x (n-1) x 2 x F
        tmp = torch.flatten(tmp, start_dim=1, end_dim=2) # B x 2*(n-1) x F
        h = torch.cat([h[:, 0:1, :],
                       tmp,
                       h[:, -1:, :].repeat(1,3,1)], dim=1)

        assert h.shape[1] == M.shape[1]


        coords_pool = M @ coords
        h_pool = M @ h


        edge_index = from_networkx(nx.complete_graph(coords.shape[1])).edge_index.to(self.device)
        edge_index_unbatch = edge_index.unsqueeze(0).repeat((batch[-1] + 1), 1, 1)


        row, col = torch.where(M > 0)
        index = torch.vstack((row + aug_size, col))
        index = torch.cat((index, torch.flip(index, dims=[0])), dim=1)
        edge_index_unbatch = torch.cat((edge_index_unbatch, index.unsqueeze(0).repeat(edge_index_unbatch.shape[0], 1, 1)), dim=2)  # B x 2 x num_edges
        h = torch.cat((h, h_pool), dim=1)
        coords = torch.cat((coords, coords_pool), dim=1)

        datalist = []
        for i in range(batch[-1] + 1):
            datalist.append(Data(edge_index=edge_index_unbatch[i]))

        # perform egnn
        data = Batch.from_data_list(datalist).to(self.device)

        h = h.view(-1, self.hidden_dim)
        coords = coords.view(-1, 3)
        row, col = data.edge_index
        out = torch.cat([h[row], h[col]], dim=1)
        edge_attr = self.edge_mlp(out)
        edge_attr = self.bn_edge(edge_attr)
        h, coords = self.egnnse3(h, coords, data.edge_index, edge_attr, data.batch)
        h = self.bn_h(h)


        # keep pooling node
        h = h.view(batch[-1] + 1, -1, h.shape[1])
        h_pool = h[:, aug_size:, :].reshape(-1, h.shape[2])
        coords = coords.view(batch[-1] + 1, -1, coords.shape[1])
        coords_pool = coords[:, aug_size:, :].reshape(-1, coords.shape[2])


        return h_pool, coords_pool


class Encoder(torch.nn.Module):
    def __init__(self, device='cpu', n_feat=1, hidden_dim=32, out_node_dim=32, in_edge_dim=32, max_length=256, layers=1,
                 egnn_layers=4, pooling=True, residual=True, attn=False, stride=2, kernel=3, padding=1):
        super(Encoder, self).__init__()

        self.max_length = max_length
        self.out_node_dim = out_node_dim
        self.layers = layers
        self.pooling = pooling
        self.device = device


        # Initialize EGNN
        if self.pooling:
            self.poolings = nn.ModuleList()
        for i in range(self.layers):

            if self.pooling:
                self.poolings.append(
                    EGNNPooling(hidden_dim=hidden_dim, stride=stride, kernel=kernel, padding=padding, attn=attn, device=device) # original is 2 2 0
                )

        if self.pooling:
            self.bn_pool = torch.nn.ModuleList([LayerNorm(in_channels=hidden_dim, mode="node") for i in range(self.layers)])
        if not self.pooling:
            self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
            self.bn_edge = LayerNorm(in_channels=hidden_dim, mode="node")

    def forward(self, coords, h, edge_index, batch, batched_data):

        # EGNN
        for i in range(self.layers):

            if not self.pooling:
                row, col = edge_index
                out = torch.cat([h[row], h[col]], dim=1)
                edge_attr = self.edge_mlp(out)
                edge_attr = self.bn_edge(edge_attr)

            if self.pooling:
                if i == 0:
                    h, coords = self.poolings[i](h, coords, batched_data.batch, batched_data, edge_index)
                else:
                    h, coords = self.poolings[i](h, coords, batched_data.batch, batched_data)
                h = self.bn_pool[i](h)

        return coords, h, batched_data, edge_index

class DecoderTranspose(torch.nn.Module):
    def __init__(self, device='cpu', hidden_dim=32, ratio=2, layers=1, attn=False, out_node_dim=32, in_edge_dim=32, egnn_layers=4, residual=True):
        super(DecoderTranspose, self).__init__()

        self.hidden_dim = hidden_dim
        self.ratio = ratio
        self.layers = layers
        self.device = device

        self.unpooling = nn.ModuleList()

        for i in range(layers):

            self.unpooling.append(
                EGNNUnPooling(hidden_dim=self.hidden_dim, stride=2, kernel=3, padding=1, output_padding=1, attn=attn, device=self.device)
            )

        self.bn = torch.nn.ModuleList([LayerNorm(in_channels=hidden_dim, mode="node") for i in range(self.layers)])

    def forward(self, coords, h, batch, batched_data, edge_index=None):

        for i in range(self.layers):
            # unpooling
            h, coords = self.unpooling[i](h, coords, batch)
            h = self.bn[i](h)

        return coords, h


class ProteinAE(torch.nn.Module):
    def __init__(self, device='cpu', layers=3, mp_steps=4, num_types=27, type_dim=32, hidden_dim=32, out_node_dim=32, in_edge_dim=32,
                 output_pad_dim=1, output_res_dim=26, pooling=True, up_mlp=False, residual=True, noise=False, transpose=False, attn=False,
                 stride=2, kernel=3, padding=1):
        super(ProteinAE, self).__init__()

        self.pooling = pooling
        self.noise = noise
        self.transpose = transpose

        self.encoder = Encoder(n_feat=type_dim, hidden_dim=hidden_dim, out_node_dim=hidden_dim,
                               in_edge_dim=hidden_dim, egnn_layers=mp_steps, layers=layers, pooling=self.pooling, residual=residual, attn=attn,
                               stride=stride, kernel=kernel, padding=padding, device=device).to(device)

        self.decoder = DecoderTranspose(device=device, hidden_dim=hidden_dim, ratio=2, layers=layers, attn=attn).to(device)

        self.residue_type_embedding = torch.nn.Embedding(num_types, hidden_dim).to(device)


        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu).to(device)

        self.bn_edge = LayerNorm(in_channels=hidden_dim, mode="node").to(device)

        self.mlp_padding = MLP(hidden_dim, hidden_dim, output_pad_dim, F.relu).to(device)

        self.mlp_residue = MLP(hidden_dim, hidden_dim * 4, output_res_dim, F.relu).to(device)

        self.sigmoid = nn.Sigmoid().to(device)
        self.device = device

        # VAE
        self.mlp_mu_h = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.mlp_sigma_h = nn.Linear(hidden_dim, hidden_dim).to(device)

        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)
        self.kl_x = 0
        self.kl_h = 0

    def add_noise(self, inputs, noise_factor=2):
        noisy = inputs + torch.randn_like(inputs) * noise_factor
        return noisy

    def forward(self, batched_data):
        # forward prop

        x, coords_ca, edge_index, batch = batched_data.x, batched_data.coords_ca, batched_data.edge_index, batched_data.batch

        if self.noise:
            coords_ca = self.add_noise(coords_ca)

        h = self.residue_type_embedding(x.squeeze(1).long()).to(self.device)

        if self.pooling:
            # encoder
            emb_coords_ca, emb_h, batched_data, edge_index = self.encoder(coords_ca, h, edge_index, batch, batched_data)

            mu_h = self.mlp_mu_h(emb_h)
            sigma_h = self.mlp_sigma_h(emb_h)

            z_h = mu_h + torch.exp(sigma_h / 2) * self.N.sample(mu_h.shape)
            self.kl_h = -0.5 * (1 + sigma_h - mu_h ** 2 - torch.exp(sigma_h)).sum() / (batch[-1] + 1)

            assert z_h.shape == emb_h.shape

            # decoder
            coords_ca_pred, h = self.decoder(emb_coords_ca, z_h, batched_data.batch, batched_data)

        else:
            coords_ca_pred, h, batched_data = self.encoder(coords_ca, h, edge_index, batch, batched_data)

        # predict padding
        pad_pred = self.sigmoid(self.mlp_padding(h))

        # predict residue type
        aa_pred = self.mlp_residue(h)

        return coords_ca_pred, aa_pred, pad_pred, self.kl_x, self.kl_h
