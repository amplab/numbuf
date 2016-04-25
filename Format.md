# Definition of the Format

Design decisions:

- In general, each data structure has a schema that contains metadata like the
	dimensions of the array, structure of the array for structured arrays,
	the data types that are stored, the type of sparse matric, etc.

- Should we make it possible to support arbitrary Python types in arrays?

	* We could detect if only a small number of different objects is present
		and use Arrow's Union datatype to efficiently encode them

	* We could also have full tags for types and allow arbitrary nesting; if this
		can be made fast in the most common case of homogeneous arrays containing
		primitive objects, it might be the way to go

  * If constructing a type like Tensor from a Python type, it is the user's
    responsibility that the data does not get deallocated; if types are
    constructed from e. g. a row batch, we hold onto the smart pointer to make
    sure the memory stays alive.

Encoding Dense Arrays
---------------------

```
Struct {
	tag: String,
	metadata: Struct {
		dtype: Int64,
		dims: List[Int64]
	}
	data: Array[Type]
}
```

Encoding Structured Arrays
--------------------------

```
Struct {
	tag: String,
	metdata: Struct {
		dtype: List [
			Struct {
				name: String
				dims: List[Int64],
				dtype: Int64
			}
		]
		dims: List[Int64]
	}
	data: Array[Type]
}
```

Encoding Sparse Matrices
------------------------

- CSR Matrices:

```
Struct {
	tag: String,
	metadata: Struct {
		dtype: Int64,
		dims: List[Int64]
	}
	data: Array[Type]
	indices: Array[Int64]
}
```

- DOK Matrices:

```
Struct {
	tag: String,
	metadata: Struct {
		dtype: Int64,
		dims: List[Int64]
	}
	data: Array [
		Struct{
			i: Int64,
			j: Int64,
			elem: Type
		}
	]
}
```

- LIL format:

```
Struct {
	tag: String,
	metadata: Struct {
		dtype: Int64,
		dims: List[Int64]
	}
	data: List[List[Type]]
}
```

Encoding Graphs
---------------

This needs to be worked out
