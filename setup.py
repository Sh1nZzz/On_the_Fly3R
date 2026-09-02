from setuptools import setup, find_packages

setup(
    name='on-the-fly3r',
    version='0.1.0',
    description='Progressive online 3D reconstruction with 3D vision foundation models.',
    license='Apache-2.0',
    packages=find_packages(
        include=[
            'evaluation',
            'evaluation.*',
            'on_the_fly3r',
            'on_the_fly3r.*',
            'third_party_codes',
            'third_party_codes.*',
        ]
    ),
)
