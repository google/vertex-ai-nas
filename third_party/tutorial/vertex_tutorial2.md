# **Tutorial-2: Create search spaces**

By now you have learnt to launch one instance of your training-docker
on cloud. But before you run a full architecture search, you
need to create a search space.

In this tutorial you will:

- Learn to specify searchable options
- Understand how the NAS-service sends an architecture suggestion to a trial
- Ingest an architecture suggestion in a trial
- Create search-space using PyGlove version of Python structures
- Create search-space using PyGlove version of classes
- Learn about search-space restrictions
- [Advanced] Create search-space with architecture-connections
- [Advanced] Create search-space with variable number of layers
- [Advanced] Handle conditional search-space

The code for this tutorial is in `tutorial/tutorial2_search_spaces.py` and
`tutorial/search_spaces.py`. Note that these files mainly focus on
the search-space. So we
have removed all the training-related code and have only
kept the search-space related code here.

## Prerequisites

This tutorial only uses one local CPU and each search space example runs
only for ~1 min. There is no quota requirement.

## Learn to specify searchable options

You will use our `PyGlove` language to create search-spaces.
`PyGlove` provides two basic-constructs to specify searchable-options
in order to create a search-space:

- `pg.one_of`
- `pg.sublist_of`

The rest of the sections in this tutorial build
search-spaces using these two constructs as the **primary** building
blocks. Let us take a look at both of them one-by-one.

### pg.one_of

You will now use the `pg.one_of` construct to create a
most basic search-space which chooses a `kernel_size` for a model:

```py
_KERNEL_SIZE = [3, 5, 7, 9]


def basic_one_of_search_space():
  return pg.Dict(kernel_size=pg.one_of(_KERNEL_SIZE))
```

#### Understand how the NAS-service sends an architecture suggestion to a trial

In `Tutorial-1`, you used the following command to launch a
baseline training:

```sh
python3 vertex_nas_cli.py search \
--project_id=${PROJECT} \
--job_name="${JOB_NAME}" \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--search_space_module=search_spaces.mnasnet_search_space \
--accelerator_type="" \
--nas_target_reward_metric="top_1_accuracy" \
--root_output_dir=${GCS_ROOT_DIR} \
--max_nas_trial=1 \
--max_parallel_nas_trial=1 \
--max_failed_nas_trial=1 \
--search_docker_flags \
num_epochs=2
```

You passed a flag `search_space_module` to the nas-client. This
was a placeholder at that time. To use the simple kernel-size
search-space, you will now pass the
`tutorial.search_spaces.basic_one_of_search_space` module to the
`search_space_module` flag instead. When the search-job is launched,
the NAS-service will spawn multiple trials (docker-copies) and pass
a different search-space-sample to each trial via the flag
`nas_params_str`. For this example, trial-1 may receive
a search-space-sample
`kernel_size=3`, trial-2 may receive a search-space-sample
`kernel_size=7`, and so on.

#### Ingest an architecture suggestion in a trial

Next we will ingest the `nas_params_str`
model-suggestion sent by the NAS-service into our code. In this tutorial,
we will not build a model using this architecture suggestion
because this tutorial is only about creating search spaces.
The following code ingests `nas_params_str` by using `cloud_nas_utils`
as a helper library and generates a search-space-sample:

```py
import cloud_nas_utils
from tutorial import search_spaces


# Get search-space from search-space-name.
search_space = search_spaces.get_search_space(argv.search_space)
# Process nas_params_str passed by the NAS-service.
# This gives one instance of the search-space to be used for this trial.
tunable_object = cloud_nas_utils.parse_and_save_nas_params_str(
    search_space=search_space,
    nas_params_str=argv.nas_params_str,
    model_dir=argv.job_dir)
unused_serialized_tunable_object = cloud_nas_utils.serialize_and_save_tunable_object(
    tunable_object=tunable_object, model_dir=argv.job_dir)
```

---
**NOTE:** The `nas_params_str` indexes into the search-space
supplied by the user.
It needs to be combined with the *search-space* to generate the final
search-space-sample `tunable_object`. Since we will cover many search-spaces
in this tutorial, we need to tell the trainer-code about which
search-space is used so that it can properly decipher the `nas_params_str`.
We use `search_space` flag and the `search_space` function to choose
the correct search-space.
Also, it is a good practice to serialize
the `tunable_object` and save to the job-directory as well.

---

---
**NOTE:** The `cloud_nas_utils` uses `logging` library to log
`tunable_object` values. We will use

```py
cloud_nas_utils.setup_logging()
```

to set the correct logging level in our main file.

---

---
**NOTE:** The `tunable_object` is a model-specification which
can be used to build a model for this trial.

---


It is a good idea to test this search-space by running the code
locally a few (2-3) times and verifying that different runs generate
different search-space-samples:

```sh
PROJECT=<Set your project-id>
REGION=<Set region for artifact registry>

# Set a unique docker-id below. It is a good practice to add your user-name
# to prevent overwriting another user's docker image.
DATE="$(date '+%Y%m%d_%H%M%S')"
TUTORIAL_DOCKER_ID=${USER}_tutorial2_${DATE}

# Build Tutorial Trainer
python3 vertex_nas_cli.py build --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--trainer_docker_file=tutorial/tutorial2.Dockerfile \
--region=${REGION}


JOB_DIR=/tmp/nas_tutorial
rm -r -f ${JOB_DIR} && mkdir ${JOB_DIR}

python3 vertex_nas_cli.py search_in_local --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--region=${REGION} \
--search_space_module=tutorial.search_spaces.basic_one_of_search_space \
--local_output_dir=${JOB_DIR} \
--search_docker_flags \
search_space='basic_one_of_search_space'
```

The multiple runs should show you different values of
`nas_params_str` and `serialized_tunable_object` each time:

```sh
nas_params_str is {"kernel_size": "3/4 (9)"}
serialized_tunable_object is {
  "kernel_size": 9
}
```

Another run:

```sh
nas_params_str is {"kernel_size": "2/4 (7)"}
serialized_tunable_object is {
  "kernel_size": 7
}
```

### pg.sublist_of

You will now use the `pg.sublist_of` construct to create a
search space consisting of all pairs of **distinct** kernel-sizes:

```py
def basic_sublist_of_search_space():
  return pg.sublist_of(
      2, _KERNEL_LIST, choices_sorted=False, choices_distinct=True)
```

Note that the first argument is 2 for the number-of-kernel-sizes-to-select and
the `choices_distinct` argument is set to `True`.

Run the following commands to print a sample from this
search-space:

```sh
# Build Tutorial Trainer
python3 vertex_nas_cli.py build --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--trainer_docker_file=tutorial/tutorial2.Dockerfile \
--region=${REGION}

JOB_DIR=/tmp/nas_tutorial
rm -r -f ${JOB_DIR} && mkdir ${JOB_DIR}

python3 vertex_nas_cli.py search_in_local --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--region=${REGION} \
--search_space_module=tutorial.search_spaces.basic_sublist_of_search_space \
--local_output_dir=${JOB_DIR} \
--search_docker_flags \
search_space='basic_sublist_of_search_space'
```

The output should look like:

```sh
nas_params_str is {"[0]": "1/4 (5)", "[1]": "2/4 (7)"}
serialized_tunable_object is [
  5,
  7
]
```

If you run the command multiple times, you will see different pairs
of distinct kernel-sizes being sampled.

As further exercise, you can try to change the number-of-kernel-sizes
from 2 to 3 and/or flip the
`choices_sorted` and `choices_distinct` arguments.

These were the most basic search-spaces. You are now ready to create
medium complexity level search-spaces using these constructs.

## Create search-space using PyGlove version of Python structures

The mnist-trainer used [a model with 2 convolutions](https://source.cloud.google.com/cloud-nas-260507/nas-codes-release/+/master:tutorial/mnist_train.py;l=64-67)
of kernel-sizes 3 each and number-of-filters as 4 and 8.
Let us try to search over possible choices of kernel-size and
number-of-filters for each of the two convolution layers. For kernel-size,
we will search between options of 3 and 5. For number-of-filters, we
try to scale them between factors of 0.75, 1.0, and 1.25.

Before you start creating this search-space using the `PyGlove` language,
it will be easier to first think about how you would define
a sample from this search-space using simple Python structures. In this
case, a sample would be a *list* of 2 "Layers" where a "Layer" is
a *dictionary* containing two values "kernel_size" and "filter_size":

```py
# NOTE: This code is just for illustration.

def search_space_sample():
  """Returns a search-space sample specification."""
  def _layer(base_filter_size):
    """Returns a layer specification."""
    return {'kernel_size': 3, 'filter_size': base_filter_size}

  # Our model has two-layers with different filter-sizes.
  return [_layer(base_filter_size=4), _layer(base_filter_size=8)]
```

The above example has *fixed/static* values for the `kernel_size`
and `filter_size`. But we want to create a search-space with
multiple options for these values. You would now
use `PyGlove` to achieve this. The reason we use
`PyGlove` language is because you can then use its
`one_of(<list of values to search>)` construct to add your search options.
**All you have to do this to
use `PyGlove` version of the list and dictionary and add `one_of`
construct to add searchable-options.** Here is our search-space:

```py
def mnist_list_of_dictionary_search_space():
  """Returns a list and dictionary based search space."""

  def _layer(base_filter_size):
    """Returns a layer specification."""
    return pg.Dict(
        kernel_size=pg.one_of([3, 5]),
        filter_size=pg.one_of(
            [int(base_filter_size * 0.75), base_filter_size,
             int(base_filter_size * 1.25)]))

  return pg.List([_layer(base_filter_size=4), _layer(base_filter_size=8)])
```

Run the following command to print out the
search-space samples on your local machine:

```sh
# Build Tutorial Trainer
python3 vertex_nas_cli.py build --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--trainer_docker_file=tutorial/tutorial2.Dockerfile \
--region=${REGION}

JOB_DIR=/tmp/nas_tutorial
rm -r -f ${JOB_DIR} && mkdir ${JOB_DIR}

python3 vertex_nas_cli.py search_in_local --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--region=${REGION} \
--search_space_module=tutorial.search_spaces.mnist_list_of_dictionary_search_space \
--local_output_dir=${JOB_DIR} \
--search_docker_flags \
search_space='mnist_list_of_dictionary_search_space'
```

You should see an output like:

```sh
nas_params_str is {"[0].kernel_size": "0/2 (3)", "[0].filter_size": "0/3 (3)", "[1].kernel_size": "1/2 (5)", "[1].filter_size": "1/3 (8)"}
serialized_tunable_object is [
  {
    "kernel_size": 3,
    "filter_size": 3
  },
  {
    "kernel_size": 5,
    "filter_size": 8
  }
]
```

You can run the command multiple times to see different values
getting sampled.

## Create search-space using PyGlove version of classes

In the previous section, note that the
`mnist_list_of_dictionary_search_space` function
eventually returns a `PyGlove` List object which in turn contains
`PyGlove` objects which in turn contain `one_of` options.
Although this works fine, but a larger/complex search-space may have
multiple nested structures and not just two simple `List` and `Dict`
structures. So it is better to have `names` for these nested structures.
You can achieve this by using classes as
building-blocks with meaningful names to build your
search-space. Or you may already have a Python class but you want
to add searchable options to it
using `PyGlove`. To learn this, let us first create a
Python *class-version* of the `search_space_sample`
and then we will make it searchable using `PyGlove`:

```py
class MyLayer:
  """Specification of a model-layer without searchable options."""

  def __init__(self, kernel_size, filter_size):
    self.kernel_size = kernel_size
    self.filter_size = filter_size


class MyModel:
  """Specification of a model with multiple layers without searchable options."""

  def __init__(self):
    self.layers = [
        MyLayer(kernel_size=3, filter_size=4),
        MyLayer(kernel_size=5, filter_size=8)
    ]
```

Note that the `Dict` has been replaced with the `MyLayer` class
and the `List` has been replaced with the `MyModel` class.
But these classes are still not searchable. Now let us
create a `PyGlove` version of these classes with
searchable options:

```py
@pg.members([('kernel_size', pg.typing.Int()),
             ('filter_size', pg.typing.Int())])
class LayerSpec(pg.Object):
  """Specification of a model-layer."""


@pg.members([('layers', pg.typing.List(pg.typing.Object(LayerSpec),
                                       min_size=1))])
class ModelSpec(pg.Object):
  """Specification of a model."""


def mnist_class_search_space():
  """Returns a class-based mnist search space."""

  def _layer(base_filter_size):
    """Returns a layer specification."""
    return LayerSpec(
        kernel_size=pg.one_of([3, 5]),
        filter_size=pg.one_of(
            [int(base_filter_size * 0.75), base_filter_size,
             int(base_filter_size * 1.25)]))

  return ModelSpec(
      layers=[_layer(base_filter_size=4),
              _layer(base_filter_size=8)])
```

To `PyGlove` a class, we derive it from `pg.Object`
class and decorate it with `@pg.members`
where the members themselves
are of [`PyGlove` type](https://cloud.google.com/vertex-ai/docs/neural-architecture-search/pyglove#constraints_for_pyglove_types).
Search-options can be added using
either `pg.one_of` or `pg.sublist_of`
constructs.

---
**NOTE:**
The topmost `PyGlove` object representing the search-space
(in this case `ModelSpec`) contains the entire nested-hierarchy of `PyGlove` objects.

---

Run the following command to print out the
search-space samples on your local machine:

```sh
# Build Tutorial Trainer
python3 vertex_nas_cli.py build --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--trainer_docker_file=tutorial/tutorial2.Dockerfile \
--region=${REGION}

JOB_DIR=/tmp/nas_tutorial
rm -r -f ${JOB_DIR} && mkdir ${JOB_DIR}

python3 vertex_nas_cli.py search_in_local --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--region=${REGION} \
--search_space_module=tutorial.search_spaces.mnist_class_search_space \
--local_output_dir=${JOB_DIR} \
--search_docker_flags \
search_space='mnist_class_search_space'
```

You should see a search-space sample output like:

```sh
nas_params_str is {"layers[0].kernel_size": "1/2 (5)", "layers[0].filter_size": "2/3 (5)", "layers[1].kernel_size": "1/2 (5)", "layers[1].filter_size": "1/3 (8)"}
serialized_tunable_object is {
  "_type": "tutorial.search_spaces.ModelSpec",
  "layers": [
    {
      "_type": "tutorial.search_spaces.LayerSpec",
      "kernel_size": 5,
      "filter_size": 5
    },
    {
      "_type": "tutorial.search_spaces.LayerSpec",
      "kernel_size": 5,
      "filter_size": 8
    }
  ]
}
```

If you run the command multiple times, then you will see different
sampled values getting printed.

## Learn about search-space restrictions

Finally, you should know about the two search-space restrictions:

- As of now, the NAS-service does not support search over continuous values.
- We recommend that the search-space-size be less than or equal to 10^20.
  This is not a hard limit. You can try using larger search space too but
  make sure that the good models do not become too sparse as the search space
  size increases even further. One possible way to deal with very large
  search spaces is to break it into two or more parts. For example, you may
  search for encoder and decoder separately one by one rather than together.
- Each instance of `pg.one_of` should ideally have <= 10 choices. Having
  more choices reduces the sampling efficiency because candidates
  under the same `pg.one_of` cannot be sampled in the same trial. This is
  not a hard limit but more of a guideline.

You now have all the tools
to build any complex search space. Going through the
`Advanced` sections of this tutorial below is optional. But
it can help you build more complicated search spaces.

## [Advanced] Create search-space with architecture-connections

Apart from choosing different parameters for a
layer/block of a model, another common
use case is about searching for different connections
between layers/blocks to create a model. You will now
create such a search space.

Let us start with ten layers indexed from `layer-0` to `layer-9`
as building blocks. We will use `layer-0` as a fixed input to `layer-1`.
But for all the subsequent layers, we will use
a combination of **any two previous layers** as an input.
For example, `layer-5` *may use* a combination of
`layer-3` and `layer-1`. Does this sound similar to what you
did earlier with choosing a pair of fruits with the
`pg.sublist_of` operation? With this in mind, let us write
the search-space definition where `layer-k` can use any two of
the `[layer-0, ... , layer-(k-1)]` as an input combination:

```py
@pg.members([('current_layer_idx', pg.typing.Int()),
             ('previous_layers', pg.typing.List(pg.typing.Int()))])
class LayerConnectionSpec(pg.Object):
  """Specifies a choice of two previous layers as an input-combination for the current layer."""


@pg.members([('layer_specs',
              pg.typing.List(pg.typing.Object(LayerConnectionSpec)))])
class ModelConnectionSpec(pg.Object):
  """Specifies connection choices for all the layers."""


def architecture_connection_search_space():
  """Returns an architecture-connection search space."""
  def _build_layer_spec(current_layer_idx):
    # Pick any 2 distinct layer-indices from the previous layers.
    previous_layers = pg.sublist_of(
        2,
        list(range(current_layer_idx)),
        choices_distinct=True,
        choices_sorted=True)
    return LayerConnectionSpec(
        current_layer_idx=current_layer_idx, previous_layers=previous_layers)

  # Build layer-specs for layers 2-to-9.
  return ModelConnectionSpec(
      layer_specs=[_build_layer_spec(i) for i in range(2, 10)])
```

Run the following command to print a sample from this search-space:

```sh
# Build Tutorial Trainer
python3 vertex_nas_cli.py build --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--trainer_docker_file=tutorial/tutorial2.Dockerfile \
--region=${REGION}

JOB_DIR=/tmp/nas_tutorial
rm -r -f ${JOB_DIR} && mkdir ${JOB_DIR}

python3 vertex_nas_cli.py search_in_local --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--region=${REGION} \
--search_space_module=tutorial.search_spaces.architecture_connection_search_space \
--local_output_dir=${JOB_DIR} \
--search_docker_flags \
search_space='architecture_connection_search_space'
```

You should see an output like:

```sh
nas_params_str is {"layer_specs[0].previous_layers[0]": "0/2 (0)", "layer_specs[0].previous_layers[1]": "1/2 (1)", "layer_specs[1].previous_layers[0]": "1/3 (1)", "layer_specs[1].previous_layers[1]": "2/3 (2)", "layer_specs[2].previous_layers[0]": "0/4 (0)", "layer_specs[2].previous_layers[1]": "1/4 (1)", "layer_specs[3].previous_layers[0]": "1/5 (1)", "layer_specs[3].previous_layers[1]": "4/5 (4)", "layer_specs[4].previous_layers[0]": "2/6 (2)", "layer_specs[4].previous_layers[1]": "5/6 (5)", "layer_specs[5].previous_layers[0]": "0/7 (0)", "layer_specs[5].previous_layers[1]": "4/7 (4)", "layer_specs[6].previous_layers[0]": "0/8 (0)", "layer_specs[6].previous_layers[1]": "2/8 (2)", "layer_specs[7].previous_layers[0]": "3/9 (3)", "layer_specs[7].previous_layers[1]": "6/9 (6)"}
serialized_tunable_object is {
  "_type": "tutorial.search_spaces.ModelConnectionSpec",
  "layer_specs": [
    {
      "_type": "tutorial.search_spaces.LayerConnectionSpec",
      "current_layer_idx": 2,
      "previous_layers": [
        0,
        1
      ]
    },
    {
      "_type": "tutorial.search_spaces.LayerConnectionSpec",
      "current_layer_idx": 3,
      "previous_layers": [
        1,
        2
      ]
    },
    {
      "_type": "tutorial.search_spaces.LayerConnectionSpec",
      "current_layer_idx": 4,
      "previous_layers": [
        0,
        1
      ]
    },
    {
      "_type": "tutorial.search_spaces.LayerConnectionSpec",
      "current_layer_idx": 5,
      "previous_layers": [
        1,
        4
      ]
    },
    {
      "_type": "tutorial.search_spaces.LayerConnectionSpec",
      "current_layer_idx": 6,
      "previous_layers": [
        2,
        5
      ]
    },
    {
      "_type": "tutorial.search_spaces.LayerConnectionSpec",
      "current_layer_idx": 7,
      "previous_layers": [
        0,
        4
      ]
    },
    {
      "_type": "tutorial.search_spaces.LayerConnectionSpec",
      "current_layer_idx": 8,
      "previous_layers": [
        0,
        2
      ]
    },
    {
      "_type": "tutorial.search_spaces.LayerConnectionSpec",
      "current_layer_idx": 9,
      "previous_layers": [
        3,
        6
      ]
    }
  ]
}
```

## [Advanced] Create search-space with variable number of layers

For the mnist-search-space, you created a 2 layer model where each layer
had independent choice of kernel-size and filter-size. But
what if we also want to choose the number-of-layers
between 2, 3, and 4 instead of just 2?

The variable length model can be thought of in terms of "skip-connection".
If a layer is marked to be skipped, then during model building
you do not add this layer to the model. Instead you forward the input
bypassing the skipped-layer:


```py

@pg.members([('kernel_size', pg.typing.Int()),
             ('filter_size', pg.typing.Int()),
             ('skip_layer', pg.typing.Bool())])
class SkippableLayerSpec(pg.Object):
  """Specifies a model-layer which can be skipped."""


@pg.members([('layer_specs',
              pg.typing.List(pg.typing.Object(SkippableLayerSpec)))])
class VariableLengthModelSpec(pg.Object):
  """Specifies a model with variable number of layers, each with different spec."""


def variable_length_model_search_space():
  """Returns a variable length model search space."""
  # To create a variable length model search space for 2, 3, or 4 layers model:
  # Create 2 layers which can not be skipped followed
  # by next 2 layers which may or may not be skipped.
  layer_specs = [
      SkippableLayerSpec(
          kernel_size=pg.oneof([3, 5]),
          filter_size=pg.oneof([4, 6, 8]),
          skip_layer=False)
  ] * 2 + [
      SkippableLayerSpec(
          kernel_size=pg.oneof([3, 5]),
          filter_size=pg.oneof([4, 6, 8]),
          skip_layer=pg.oneof([True, False]))
  ] * 2
  return VariableLengthModelSpec(layer_specs=layer_specs)

```

You can run the following command to sample this search-space:

```sh
# Build Tutorial Trainer
python3 vertex_nas_cli.py build --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--trainer_docker_file=tutorial/tutorial2.Dockerfile \
--region=${REGION}

JOB_DIR=/tmp/nas_tutorial
rm -r -f ${JOB_DIR} && mkdir ${JOB_DIR}

python3 vertex_nas_cli.py search_in_local --project_id=${PROJECT} \
--trainer_docker_id=${TUTORIAL_DOCKER_ID} \
--region=${REGION} \
--search_space_module=tutorial.search_spaces.variable_length_model_search_space \
--local_output_dir=${JOB_DIR} \
--search_docker_flags \
search_space='variable_length_model_search_space'
```

This should generate an output similar to:

```sh
nas_params_str is {"layer_specs[0].kernel_size": "0/2 (3)", "layer_specs[0].filter_size": "2/3 (8)", "layer_specs[1].kernel_size": "0/2 (3)", "layer_specs[1].filter_size": "2/3 (8)", "layer_specs[2].kernel_size": "0/2 (3)", "layer_specs[2].filter_size": "0/3 (4)", "layer_specs[2].skip_layer": "1/2 (False)", "layer_specs[3].kernel_size": "1/2 (5)", "layer_specs[3].filter_size": "1/3 (6)", "layer_specs[3].skip_layer": "0/2 (True)"}
serialized_tunable_object is {
  "_type": "tutorial.search_spaces.VariableLengthModelSpec",
  "layer_specs": [
    {
      "_type": "tutorial.search_spaces.SkippableLayerSpec",
      "kernel_size": 3,
      "filter_size": 8,
      "skip_layer": false
    },
    {
      "_type": "tutorial.search_spaces.SkippableLayerSpec",
      "kernel_size": 3,
      "filter_size": 8,
      "skip_layer": false
    },
    {
      "_type": "tutorial.search_spaces.SkippableLayerSpec",
      "kernel_size": 3,
      "filter_size": 4,
      "skip_layer": false
    },
    {
      "_type": "tutorial.search_spaces.SkippableLayerSpec",
      "kernel_size": 5,
      "filter_size": 6,
      "skip_layer": true
    }
  ]
}
```

For the above example, the last layer is marked to be skipped.
When building the model using this spec, you will not use the last layer
and thus create a 3 layer model.

## [Advanced] Handle conditional search space

As a corner case, sometimes a certain combination of search-space
options may be invalid. Let us say you have `kernel_size` that
can take values 1, 2 and 3 and `filter_size` that can take
values 8, 9, 10, but {`kernel_size`=2 and `filter_size`=9}
should be treated as an invalid combination.

Our recommendation is that if the number of such invalid cases are below
~ 20% of the search-space, then
one option is to allow such combinations in the search-space,
but get the training code to fail
for this invalid combination by not returning a reward back.

However, if the number of such invalid combinations are
more than ~ 20% of the search-space, then you should rethink
the search-space by decoupling the dependent variables.
For example, let us say that you have two search-space
variables `x` and `y`,
and you want to constrain their sampling with the formula
`-1 <= x + y <= 1`. In this case `x` and `y` are coupled with
a dependency. You can not directly create a search-space like:

```py
@pg.members([('x', pg.typing.Int()),
             ('y', pg.typing.Int())])
class XYSpec(pg.Object):
  """Specifies a choice of X and Y."""

def conditional_search_space():
  return XYSpec(x=pg.one_of(list(range(-10, 11))), y=pg.one_of(list(range(-10, 11))))
```

because it can sample values such as `x = 5` and `y = 5` which do not
satisfy the constraint `-1 <= x + y <= 1`.
To decouple the search-space, you can
reformulate this constraint
into `y = z - x` where `-1 <= z <= 1`. This way you create an equivalent
search-space over `x` and `z` which are decoupled and still
achieve the same goal:

```py
@pg.members([('x', pg.typing.Int()),
             ('z', pg.typing.Int())])
class XZSpec(pg.Object):
  """Specifies a choice of X and Z to decouple a conditional search-space."""


def conditional_search_space():
  """Creates a conditional search-space.

  Let us say that you have two search-space variables `x` and `y`,
  and you want to constrain their sampling with the formula
  `-1 <= x + y <= 1`. In this case `x` and `y` are coupled with a dependency.
  To decouple the search-space, you can reformulate this constraint
  into `y = z - x` where `-1 <= z <= 1`.

  Returns:
    A conditional search-space.
  """
  return XZSpec(
      x=pg.one_of(list(range(-10, 11))), z=pg.one_of([-1, 0, 1]))
```

Use `y = z-x` in your training code, when the NAS-services sends you
a sample of `x` and `z`.
