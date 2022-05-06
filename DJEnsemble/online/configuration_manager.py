class ConfigurationManager:
    configuration_dictionary = {}

    def __init__(self, configuration_file_path):
        self.read_configuration_file(configuration_file_path)

    def read_configuration_file(self, configuration_file_path):
        file = open(configuration_file_path)
        for configuration in file:
            if configuration[0].strip() == '#' or configuration.strip() == '':
                continue
            option, value = tuple(configuration.split(':'))
            option, value = option.strip(), value.strip()
            self.configuration_dictionary[option] = value

    def get_configuration_value(self, configuration):
        return self.configuration_dictionary[configuration]