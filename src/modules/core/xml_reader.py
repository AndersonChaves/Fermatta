import csv
import requests
import xml.etree.ElementTree as ET

class XMLReader():
    tree = None

    def read_xml(self, url):
        self.url = 'http://alertario.rio.rj.gov.br/upload/xml/Chuvas.xml'
        resp = requests.get(url)
        with open('alerta-rio.xml', 'wb') as f:
            f.write(resp.content)
        self.tree = ET.parse('alerta-rio.xml')

    def get_attribute(self, path, attribute_name):
        root = self.tree.getroot()
        station_nodes_list = root.findall('./estacao')
        for child in station_nodes_list:
            if child.tag == 'estacao':
                if child.attrib['nome'] == 'Rocinha':
                  items = child.findall('./chuvas')
                  for child in items:
                    if child.tag == 'chuvas':
                      return child.attrib['mes']

    def get_rain_for_station(self, station_name, variable):
        root = self.tree.getroot()
        station_nodes_list = root.findall('./estacao')
        for child in station_nodes_list:
            if child.tag == 'estacao':
                if child.attrib['nome'] == station_name:
                  items = child.findall('./chuvas')
                  for child in items:
                    if child.tag == 'chuvas':
                      return child.attrib[variable]

    def get_record_time(self):
        root = self.tree.getroot()
        return root.attrib['hora']

# Example
#x = XMLReader()
#x.read_xml('http://alertario.rio.rj.gov.br/upload/xml/Chuvas.xml')
#print(x.get_record_time())
#print(type(x.get_record_time()))