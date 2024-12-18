import tempfile
from pm4py.util import exec_utils, constants
from enum import Enum
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from typing import Optional, Dict, Any, Union, List
import graphviz


class Parameters(Enum):
    FORMAT = "format"
    ENABLE_GRAPH_TITLE = "enable_graph_title"
    GRAPH_TITLE = "graph_title"


def apply(clf: DecisionTreeClassifier, feature_names: List[str], classes: List[str], parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> graphviz.Source:
    """
    Apply the visualization of the decision tree

    Parameters
    ------------
    clf
        Decision tree
    feature_names
        Names of the provided features
    classes
        Names of the target classes
    parameters
        Possible parameters of the algorithm, including:
            Parameters.FORMAT -> Image format (pdf, svg, png ...)

    Returns
    ------------
    gviz
        GraphViz object
    """
    if parameters is None:
        parameters = {}

    enable_graph_title = exec_utils.get_param_value(Parameters.ENABLE_GRAPH_TITLE, parameters, constants.DEFAULT_ENABLE_GRAPH_TITLES)
    graph_title = exec_utils.get_param_value(Parameters.GRAPH_TITLE, parameters, "Decision Tree")

    format = exec_utils.get_param_value(Parameters.FORMAT, parameters, "png")
    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    filename.close()

    dot_data = export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=classes,
                                    filled=True, rounded=True,
                                    special_characters=True)

    if enable_graph_title:
        dot_data = dot_data.replace('digraph Tree {',
                                               f'digraph Tree {{\ngraph [label="{graph_title}", labelloc=t, fontsize=20];')

    gviz = graphviz.Source(dot_data)
    gviz.format = format
    gviz.filename = filename.name

    return gviz
