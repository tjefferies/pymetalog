pymetalog
================
Colin Smith, Travis Jefferies, Isaac J. Faber

> [!IMPORTANT]
> **This repository has been archived and is no longer maintained.**
>
> 👉 **Please visit the new repository: [metalog_jax](https://github.com/tjefferies/metalog_jax)**
> 
> **Issues and discussions on this repo are not monitored.**
>
> **Note:** The new repository is not a direct continuation of this project. It differs significantly in architecture / API / approach. If you're starting fresh, we recommend the new repo. If you depend on this codebase, be aware that ***no updates or fixes will be made here***.
>
> This repo remains available for historical reference only.
---

`pip install pymetalog`

### The Python Metalog Distribution

This repo is a working project for a python package (**pymetalog**) that generates functions
for the metalog distribution. The metalog distribution is a highly
flexible probability distribution that can be used to model data without
traditional parameters.

### Metalog Background

In economics, business, engineering, science and other fields,
continuous uncertainties frequently arise that are not easily- or
well-characterized by previously-named continuous probability
distributions. Frequently, there is data available from measurements,
assessments, derivations, simulations or other sources that characterize
the range of an uncertainty. But the underlying process that generated
this data is either unknown or fails to lend itself to convenient
derivation of equations that appropriately characterize the probability
density (PDF), cumulative (CDF) or quantile distribution functions.

The metalog distributions are a family of continuous univariate
probability distributions that directly address this need. They can be
used in most any situation in which CDF data is known and a flexible,
simple, and easy-to-use continuous probability distribution is needed to
represent that data. Consider their [uses and
benefits](http://www.metalogdistributions.com/usesbenefits.html). Also
consider their
[applications](http://www.metalogdistributions.com/applicationsdata.html)
over a wide range of fields and data sources.

This repository is a complement and extension of the information found
in the [paper
published](http://pubsonline.informs.org/doi/abs/10.1287/deca.2016.0338)
in Decision Analysis and the
[website](http://www.metalogdistributions.com/)
