# OpenVINO IREE Plugin

## Description

This is a plugin which uses [IREE](https://iree.dev) compiler and runtime for model preparing and inference on a system

## Build

To enable the plugin you need to use additional cmake arguments:
* -DENABLE_IREE=ON - enables building IREE plugin
* -DENABLE_IREE_SRC=/path/to/iree/ - need to provide path to IREE's sources
* -DENABLE_IREE_LIB=/path/to/iree-build/ - need to provide path to pre-built IREE's libs

## Components

IREE Plugin contains the following components:

* [src](./src/) - sources of the plugin.
* [include](./include) - headers of the plugin.

## See also

 * [OpenVINO™ README](../../../README.md)
 * [OpenVINO Core Components](../../README.md)
 * [OpenVINO Plugins](../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
