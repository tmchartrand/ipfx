"""
"""
import copy as cp
import io
from typing import (
    List, Dict, Any, Set, Optional
)

import yaml

from ipfx.attach_metadata.sink.metadata_sink import (
    MetadataSink, OneOrMany
)


class DandiYamlSink(MetadataSink):
    """ Sink specialized for writing data to a DANDI-compatible YAML file.
    """

    @property
    def targets(self) -> List[Dict[str, Any]]:
        return self._targets

    @property
    def supported_cell_fields(self) -> Set[str]:
        return {
            "species",
            "age",
            "sex",
            "gender",
            "date_of_birth",
            "genotype",
            "cre_line"
        }

    @property
    def supported_sweep_fields(self) -> Set[str]:
        return set()

    def __init__(self):

        self._targets: List[Dict] = []
        self._data = {}

    def serialize(self, targets: Optional[OneOrMany[Dict[str, Any]]] = None):
        """ Write this sink's data to one or more 
        """

        for target in self._ensure_plural_targets(targets):
            target = cp.deepcopy(target)
            if not isinstance(target["stream"], io.IOBase):
                target["stream"] = open(target["stream"], "w")

            yaml.dump(self._data, **target)
            target["stream"].close()

    def register(
            self, 
            name: str, 
            value: Any, 
            sweep_id: Optional[int] = None
    ):
        """ Attaches a named piece of metadata to this sink's internal store. 
        Should dispatch to a protected method which carries out appropriate 
        validations and transformations.

        Parameters
        ----------
        name : the well-known name of the metadata
        value : the value of the metadata (before any required transformations)
        sweep_id : If provided, this will be interpreted as sweep-level 
            metadata and sweep_id will be used to identify the sweep to which 
            value ought to be attached. If None, this will be interpreted as 
            cell-level metadata

        Raises
        ------
        ValueError : An argued piece of metadata is not supported by this sink
        """

        if name in self.supported_cell_fields:
            # this format is just a straightforward mapping
            self._data[name] = value

        else:
            raise ValueError(
                f"don't know how to attach metadata field: {name}\n"
            )