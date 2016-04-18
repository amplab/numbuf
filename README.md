# Ether: Powering In-Memory Scientific Data Analysis

Ether is an in-memory representation for scientific data (including Computer
Vision, Graphics, Machine Learning, Optimization, Graph Analytics) based on
Apache Arrow. Arrow defines the fundamental building blocks like Structs,
Unions, Arrays, List and primitive types and Ether defines how these are used
in a standardized way to describe scientific data.

A related project is the Feather project by the Arrow developers, which defines
a concrete layout for DataFrames.

Our goals are:

- Defining an in-memory layout for various scientific data structures,
	so data can be manipulated from different environments like Python,
	MATLAB, Julia, R using native data types, without serialization

- Enabling quick movement of data from storage systems like 3D XPoint to CPU
	memory, between NUMA memory banks, from CPU memory to accelerator (like GPU)
	memory, between different nodes over the network, between accelerators via
	RDMA

- Being able to write the data to disk in a compressed format (most likely
	Parquet, this should already be supported by Arrow)

- Providing an implementation of this layout that targets the scientific Python
	stack (NumPy, SciPy)

Data structures we plan to support include:

- Dense Arrays, including arbitrary strides and structured arrays
- Representations of sparse matrices like DOK, LIL, COO, CSR, CRS, CSC, CCS
- Representation for graph structured data
- Lists and Dictionaries of the above
