#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness

if (__name__ == '__main__'):
    harness = ConfigurationImporterCliHarness(
        src_dir_name='src',
        app_factory_class='zensols.amrspring.ApplicationFactory',
        proto_args='predict "Had CTA showing PE predominantly in L main PA w/ a small non-occlusive strand."',
        proto_factory_kwargs={'reload_pattern': r'^zensols.amrspring'},
    )
    harness.run()
