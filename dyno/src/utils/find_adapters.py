import pysoem
# finds all adapters on the system

adapters = pysoem.find_adapters()
for i, adapter in enumerate(adapters):
    print('Adapter {}'.format(i))
    print('  {}'.format(adapter.name))
    print('  {}'.format(adapter.desc))