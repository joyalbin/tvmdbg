# pylint: disable=unused-argument
"""Debug runtime functions."""

import json
import os
import numpy as np
from tvm import ndarray as nd
from tvm.tools.debug.wrappers import local_cli_wrapper as tvmdbg


def _ensure_dir(file_path):
    """Create a directory if not exists

    Parameters
    ----------

    file_path: str
        File path to create

    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class NodeStepper(object):
    #TVMDebug stepper Registration

    def __init__(self, cli_obj):
        #Constructor for NodeStepper
        self.next_exec = 0
        self._cli_obj = cli_obj
        self.node_exec_status = []
        self.graph_nodes_count = get_graph_node_count(cli_obj)
        for i in range(self.graph_nodes_count):
            self.node_exec_status.append(False)

    def get_exec_status(self, target_node):
        return self.node_exec_status[target_node]

    def set_exec_status(self, target_node, status):
        self.node_exec_status[target_node] = status

    def set_next_node(self, target_node):
        self.next_exec = target_node + 1

    def get_next_node(self):
        return self.next_exec

    def get_graph_nodes_count(self):
        return self.graph_nodes_count

    def get_additional_target_nodes(self, target_node):
        additional_target_nodes = []
        input_list = get_node_inputs_index(self._cli_obj, target_node)
        if input_list == None:
            print("Current node %d doesnot have an input", target_node)
            return
        exec_inp = []
        for input in input_list:
            exec_inp.append(input)
        while len(exec_inp) > 0:
            input = exec_inp.pop()
            if self.get_exec_status(input) == True:
                continue;
            additional_target_nodes.append(input)
            sub_input_list = get_node_inputs_index(self._cli_obj, input)
            for sub_input in sub_input_list:
                if self.get_exec_status(sub_input) != True \
                        and sub_input not in exec_inp:
                    exec_inp.append(sub_input)
        return additional_target_nodes

def stepper_init(cli_obj):
    print("stepper_init called")
    stepper = NodeStepper(cli_obj)
    return stepper

def get_graph_node_count(cli_obj):
    return len(cli_obj._nodes_list)

def get_node_raw_index(cli_obj, index):
    """Return the output raw node index

    Parameters
    ----------
    node_index: int
        Node index from the NNVM graph

    Returns
    ----------
    Raw node index
    """
    return cli_obj.node_row_ptr_list[index]

def get_node_inputs_index(cli_obj, index):
    """Return the input nodes index from the graph

    Parameters
    ----------
    node_index: int
        The node index for which the inputs has to find

    Returns
    ----------
    Input nodes index from the graph
    """
    node_list = cli_obj._nodes_list
    inputs_name = []
    for i in range (len(node_list)):
        if i == index:
            cur_node = node_list[i]
            if cur_node['op'] == 'param':
                break
            inputs_name = cur_node['inputs']
            break;

    inputs_id = []
    for raw_node_id in range (len(node_list)):
        node = node_list[raw_node_id]
        for input in inputs_name:
            if node['name'] == input:
                inputs_id.append(raw_node_id)
    return inputs_id

def _dump_json(ctx, cli_obj, dltype_list, shapes_list):
    """Dump the nodes in json format to file

    Parameters
    ----------

    ctx: Str
        context in string

    cli_obj: obj
        CLI object where common information is stored

    dltype_list: List
        List of datatypes of each node

    shapes_list: List
        List of shape of each node

    """

    nodes_list = cli_obj._nodes_list
    new_graph = {}
    new_graph['nodes'] = []
    nodes_len = len(nodes_list)
    for i in range(nodes_len):
        node = nodes_list[i]
        input_list = []
        for input_node in node['inputs']:
            input_list.append(nodes_list[input_node[0]]['name'])
        #del node['inputs']
        node['inputs'] = input_list
        dltype = str("type: " + dltype_list[1][i])
        if 'attrs' not in node:
            node['attrs'] = {}
            node['op'] = "param"
        else:
            node['op'] = node['attrs']['func_name']
        node['name'] = node['name'].replace("/", "_")
        node['attrs'].update({"T": dltype})
        node['shape'] = shapes_list[1][i]
        new_graph['nodes'].append(node)

    # save to file
    graph_dump_file_name = '_tvmdbg_graph_dump.json'
    folder_name = "/_tvmdbg_device_,job_localhost,replica_0,task_0,device_"
    folder_name = folder_name + ctx.replace(":", "_") + "/"
    cli_obj._dump_folder = folder_name
    path = cli_obj._dump_root + folder_name
    _ensure_dir(path)
    with open((path + graph_dump_file_name), 'w') as outfile:
        json.dump(new_graph, outfile, indent=2, sort_keys=False)


def _dump_heads(cli_obj, heads_list):
    """Dump the heads to a list

    Parameters
    ----------

    cli_obj: obj
        The CLI object

    heads_list : List
       The list of outputs from the json node

    """
    for output in heads_list:
        cli_obj.set_ouputs(cli_obj._nodes_list[output[0]]['name'])


def dump_output(cli_obj, ndarraylist):
    """Dump the outputs to a temporary folder

    Parameters
    ----------

    cli_obj: obj
        The CLI object

    ndarraylist : List of tvm.ndarray
       The list of outputs, for each node in node list

    """
    order = 1
    eid = 0
    for node in cli_obj._nodes_list:
        num_outputs = 1 if node['op'] == 'param' else int(node['attrs']['num_outputs'])
        for j in range(num_outputs):
            ndbuffer = ndarraylist[eid]
            eid = eid + 1
            key = node['name'] + "_" + str(j) + "__000000" + str(order) + ".npy"
            key = key.replace("/", "_")
            file_name = str(cli_obj._dump_root + cli_obj._dump_folder + key)
            np.save(file_name, ndbuffer.asnumpy())
            os.rename(file_name, file_name.rpartition('.')[0])
            order = order + 1


def set_input(cli_obj, key=None, value=None, **params):
    """Set inputs to the module via kwargs

    Parameters
    ----------
    cli_obj: obj
        The CLI object

    key : int or str
       The input key

    value : the input value.
       The input key

    params : dict of str to NDArray
       Additonal arguments
    """
    if key:
        cli_obj.set_input(key.replace("/", "_"), value)


def create(obj, graph):
    """Create a debug runtime environment and start the CLI

    Parameters
    ----------
    obj: Object
        The object being used to store the graph runtime.

    graph: str
        NNVM graph in json format

    """
    ctx = str(obj.ctx).upper().replace("(", ":").replace(")", "")
    cli_obj = tvmdbg.LocalCLIDebugWrapperSession(obj, graph, ctx=ctx)
    json_obj = json.loads(graph)
    cli_obj._nodes_list = json_obj['nodes']
    dltype_list = json_obj['attrs']['dltype']
    shapes_list = json_obj['attrs']['shape']
    heads_list = json_obj['heads']
    cli_obj.node_row_ptr_list = json_obj['node_row_ptr']
    # dump the json information
    _dump_json(ctx, cli_obj, dltype_list, shapes_list)
    _dump_heads(cli_obj, heads_list)
    # prepare the out shape
    obj.ndarraylist = []
    for i in range(len(shapes_list[1])):
        obj.ndarraylist.append(nd.empty(shapes_list[1][i], dltype_list[1][i]))
    return cli_obj
