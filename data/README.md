# Input Format

## intervals-AST.json
This is the input of the model in models/GGNN.py
```C++
{
    "projName":"X1", // A string to record information
    "node_mask":[false,...,true,...,false], // A unmasked node denotes it is the entry of a statement
    "fileHash":["1.java"], // A string to record information
    "bugPos":[0], // Buggy node position
    "targets":[[0.0]], // Whether it is a clean (0.0) or buggy method (1.0)
    "graph":[[1,1,2],...,[8,2,10]], // A list of edges in the form of [source_node, edge_type, target_node]
    "node_features":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],...[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], // A predefined node features.
    "funName":"" // A string to record information
}
```

## intervals.json
This is the input of the model in models/GINN.py

```C++
{
    // We preprocess a method to construct first-order intervals,
    "0":{ 
        "numOfFeatures":[1,1,1],
        "intervalID":0, // The ID of the interval
        "node_mask":[false,true,false],
        "bugPos":[0],
        "graph":[[1,1,0],[2,1,1]],
        "convRep":[[66],[43],[66]], // This is another kind of node features.
        "insideinterval":1, // It indicates whether we are processing intervals (1), or not (0)
        "node_features":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],...,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]]
        },
    "numOfNode":1,
    "projName":"X1",
    "fileHash":["1.java"],
    "bugPos":[0],
    "targets":[[0.0]],
    "graph":[[0,1,0]],
    "insideinterval":0, // It indicates whether we are processing intervals (1), or not (0)
    "funName":""
}
```


## Sandwish-intervals.json
This is the input of the model in models/Sandwiches-interval.py

```C++
{
    // We preprocess a method to construct first-order intervals,
    "0":{ 
        ...
        "node_mask":[false,true,false], // Keep one node in an AST of a statement and mask other nodes.
        ...
        },
    ...
    "targets":[Label, BugPos, FixPos], // Label: whehter the method is a buggy method or not; if the method is buggy, which variable is incorrect (i.e., BugPos), which variable is correct (FixPos).
    ...
}
```