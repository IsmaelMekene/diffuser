import ruamel.yaml

yaml = ruamel.yaml.YAML()
data = yaml.load(open('environment.yml'))

requirements = []
for dep in data['dependencies']:
    if isinstance(dep, str):
        print( dep.split('='))
        package, package_version = dep.split('=')
        #if python_version == '0':
        #    continue
        requirements.append(package + '==' + package_version)
    elif isinstance(dep, dict):
        for preq in dep.get('pip', []):
            requirements.append(preq)

with open('requirements.txt', 'w') as fp:
    for requirement in requirements:
       print(requirement, file=fp)