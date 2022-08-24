#!/usr/bin/env python3
"""Library for the analysis of omics data profiles along an organ sections."""
import os
from math import ceil
import glob
import pickle
import configparser
import shutil
import warnings
import ast
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, beta, median_abs_deviation
from scipy.optimize import curve_fit
from supervenn import supervenn
import sympy as sym
from tqdm import tqdm


class ProfileAnalysis:
    """Class for the analysis of omics data along an organ sections."""

    def __init__(self, project_path):
        """Init method of ProfileAnalysis class.

        During class creation the init method will search for a file named
        SETTINGS.ini in the path specified in the variable settings. If the
        file is found than the project is created, if not a warning is issued.

        Parameters
        ----------
        settings : string
            path were the project needs to be created.

        Returns
        -------
        Create the folder structure of the project
        """
        self.project_path = project_path
        self.settings_path = os.path.join(self.project_path, 'SETTINGS.ini')
        self.github = 'https://github.com/Dallavilla-Tiziano/profile_analysis'
        if os.path.isfile(self.settings_path):
            self.load_configuration()
            self.create_project()
            print(f'Project has been created!')
        else:
            raise FileNotFoundError("SETTINGS.ini could not be found, the "
                                    "project can't be created. Please check "
                                    f"the documentation at {self.github}")

    def load_configuration(self):
        """Read settings from SETTINGS.ini."""
        # Read file
        config = configparser.ConfigParser()
        config.read(self.settings_path)
        try:
            self.sections = ast.literal_eval(config['ORGAN']['sections'])
        except SyntaxError as syntax_error:
            raise SyntaxError('Variable sections is not a dictionary!')\
                from syntax_error
        try:
            self.sections4plots = ast.literal_eval(config['ORGAN']['plot_names'])
        except SyntaxError as syntax_error:
            raise SyntaxError('Variable plot_names is not a list!')\
                from syntax_error
        try:
            self.x = ast.literal_eval(config['ORGAN']['sections_distance_from_reference'])
        except SyntaxError as syntax_error:
            raise SyntaxError("Variable sections_distance_from_reference"
                              "is not a list!'") from syntax_error
        try:
            self.cores = int(config['MISC']['Cores'])
        except ValueError as value_error:
            raise ValueError("Variable cores is not an integer")\
                from value_error
        try:
            self.sample_0_t = float(config['ANALYSIS_SETTINGS']['Sample_0_threshold'])
        except ValueError as value_error:
            raise ValueError("Variable sample_0_t is not a float")\
                from value_error
        try:
            self.degree_2_test = int(config['ANALYSIS_SETTINGS']['polynomial_degree_to_test'])
        except ValueError as value_error:
            raise ValueError("Variable polynomial_degree_to_test is not an"
                             "integer") from value_error
        try:
            self.rnd_perm_n = int(config['ANALYSIS_SETTINGS']['random_permutation_n'])
        except ValueError as value_error:
            raise ValueError("Variable random_permutation_n is not an"
                             "integer") from value_error
        self.data_type = config['ANALYSIS_SETTINGS']['data_type']
        if self.data_type not in ['numeric', 'binary']:
            raise ValueError("data_type can only be 'numeric' or 'binary'."
                             "Please check "
                             f"the documentation at {self.github}")
        self.samples2sections = {}
        self.input_data = os.path.join(self.project_path, 'input_data')
        self.data_raw = os.path.join(self.input_data, 'raw')
        self.data_clinical = os.path.join(self.input_data, 'clinical')
        self.sample_by_section = os.path.join(self.project_path,
                                              'sample_by_section_1')
        self.data_fitting = os.path.join(self.project_path, 'data_fitting_2')
        self.rnd_data_fitting = os.path.join(self.project_path,
                                             'random_data_fitting_3')
        self.figures = os.path.join(self.project_path, 'figures')
        self.output = os.path.join(self.project_path, 'output')
        try:
            self.plot_font_size = int(config['MISC']['plot_font_size'])
        except ValueError as value_error:
            raise ValueError("Variable plot_font_size is not an"
                             "integer") from value_error
        try:
            self.t_area = float(config['ANALYSIS_SETTINGS']['threshold_area'])
        except ValueError as value_error:
            raise ValueError("Variable threshold_area is not a float")\
                from value_error
        self.index_col = config['ANALYSIS_SETTINGS']['index_col']
        matplotlib.rcParams.update({'font.size': self.plot_font_size})

    def create_project(self):
        """Create project folders structure."""
        if os.path.isdir(self.input_data):
            warnings.warn("Folder input_data already exist...skipping",
                          UserWarning)
        else:
            os.makedirs(self.input_data)

        if os.path.isdir(self.data_raw):
            warnings.warn("Folder data_raw already exist...skipping",
                          UserWarning)
        else:
            os.makedirs(self.data_raw)

        if os.path.isdir(self.data_clinical):
            warnings.warn("Folder data_clinical already exist...skipping",
                          UserWarning)
        else:
            os.makedirs(self.data_clinical)

        if os.path.isdir(self.sample_by_section):
            warnings.warn("Folder sample_by_section already exist...skipping",
                          UserWarning)
        else:
            os.makedirs(self.sample_by_section)

        if os.path.isdir(self.data_fitting):
            warnings.warn("Folder data_fitting already exist...skipping",
                          UserWarning)
        else:
            os.makedirs(self.data_fitting)

        if os.path.isdir(self.rnd_data_fitting):
            warnings.warn("Folder rnd_data_fitting already exist...skipping",
                          UserWarning)
        else:
            os.makedirs(self.rnd_data_fitting)

        if os.path.isdir(self.figures):
            warnings.warn("Folder figures already exist...skipping",
                          UserWarning)
        else:
            os.makedirs(self.figures)

        if os.path.isdir(self.output):
            warnings.warn("Folder output already exist...skipping",
                          UserWarning)
        else:
            os.makedirs(self.output)

    def print_settings(self):
        """Print the current settings."""
        print(f"Settings file path: {self.settings_path}")
        print(f"Defined sections: {self.sections}")
        print(f"Section names for plotting: {self.sections4plots}")
        print(f"Section distance from reference point: {self.x}")

        print(f"Index colon name: {self.index_col}")
        print(f"Maximum percentage of 0s to drop a sample: {self.sample_0_t}")
        print(f"Random distribution area threshold: {self.t_area}")
        print(f"Maximum polynomial degree to test: {self.degree_2_test}")
        print(f"Number of random permutations: {self.rnd_perm_n}")
        print(f"Type of data analysed: {self.data_type}")

        print(f"Input data folder: {self.input_data}")
        print(f"Raw data folder: {self.data_raw}")
        print(f"Clinical data folder: {self.data_clinical}")
        print(f"Median by section data folder: {self.sample_by_section}")
        print(f"Observable fitting folder: {self.data_fitting}")
        print(f"Random permutation fitting folder: {self.rnd_data_fitting}")
        print(f"Figure folder: {self.figures}")
        print(f"Output data folder: {self.output}")

        print(f"Plot font size: {self.plot_font_size}")
        print(f"Number of cores used: {self.cores}")
        print(f"Documentation link: {self.github}")

    def check_step_completion(self, path, pkl=0):
        """Check if a step of the pipeline has been executed."""
        result = pd.DataFrame()
        if pkl:
            if glob.glob(path):
                print('This step has already been executed...loading results...')
                result = pd.read_pickle(path)
        else:
            if glob.glob(path):
                print('This step has already been executed...loading results...')
                result = pd.read_csv(path)
        return result

    def create_samples_to_sections_table(self):
        """
        Assign each sample to a section.
        Create a class dictionary 'samples2sections' that can be accessed
        """
        # DATA
        self.clinical_data = pd.read_csv(glob.glob(f'{self.data_clinical}/*.csv')[0])
        self.data_table = pd.read_csv(glob.glob(f'{self.data_raw}/*.csv')[0])
        self.data_table.set_index(self.index_col, inplace=True)
        self.samplesPerSection = []
        for key in self.sections.keys():
            section_samples = self.clinical_data[self.clinical_data['site_of_resection_or_biopsy'].isin(self.sections[key])]['sample_submitter_id'].to_list()
            #Checking that samples in clinical data are in raw data
            section_samples = set(section_samples).intersection(self.data_table.columns)
            self.samples2sections[key] = list(section_samples)
            self.samplesPerSection.append(len(section_samples))
        sample_to_section = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in
                                         self.samples2sections.items()]))
        sample_to_section.fillna('', inplace=True)
        sample_to_section.to_csv(os.path.join(self.output,
                                 'samples_by_sections.csv'), index=False)

    def calculate_median_by_section_numeric(self, table, remove_outliers):
        """
        Calculate samples median and mad values for each section.
        Genes with a % of zero measurments above 'sample_0_t'
        (defined in settings) are discarded from the analysis.

        Parameters
        ----------
        feature : DataFrame
            row of self.data_table containing measurments for a single gene

        remove_outliers : bool
            if true remove samples outside range 'median+/-2.5*MAD' before
            calculation of median by section

        Returns
        -------
        medians_df : DataFrame
            DataFrame with median values per section. If the number of zero
            measurment is above 'sample_t_0' DataFrame will be empty.

        mad_df : DataFrame
            DataFrame with MAD for eah section.
        """
        medians_df = pd.DataFrame()
        mad_df = pd.DataFrame()
        for index, feature in table.iterrows():
            if int(feature.isin([0]).sum())/len(feature) <= self.sample_0_t:
                for section in self.sections:
                    if remove_outliers:
                        section_samples = feature[self.samples2sections[section]]
                        median = np.median(section_samples)
                        mad_inf = median - 2.5*mad
                        mad_sup = median + 2.5*mad
                        values = [i for i in feature
                                  if mad_inf < i < mad_sup]
                    else:
                        values = feature[self.samples2sections[section]]
                    median = np.median(values)
                    mad = median_abs_deviation(values)
                    medians_df.loc[index, section] = median
                    mad_df.loc[index, section] = mad
        medians_df.dropna(inplace=True)
        mad_df = mad_df.loc[medians_df.index]
        return medians_df, mad_df

    def calculate_median_by_section_binary(self, table):
        """
        Calculate samples median and mad values for each section.
        Genes with a % of zero measurments above 'sample_0_t'
        (defined in settings) are discarded from the analysis.

        Parameters
        ----------
        feature : DataFrame
            row of self.data_table containing measurments for a single gene

        remove_outliers : bool
            if true remove samples outside range 'median+/-2.5*MAD' before
            calculation of median by section

        Returns
        -------
        medians_df : DataFrame
            DataFrame with median values per section. If the number of zero
            measurment is above 'sample_t_0' DataFrame will be empty.

        mad_df : DataFrame
            DataFrame with MAD for eah section.
        """
        medians_df = pd.DataFrame()
        mad_df = pd.DataFrame()
        for index, row in table.iterrows():
            for section in self.sections:
                cnv=(row[self.samples2sections[section]]>=1).sum()
                nocnv=(row[self.samples2sections[section]]==0).sum()
                frac = cnv/(cnv+nocnv)*100
                medians_df.loc[index, section] = frac
        medians_df.dropna(inplace=True)
        mad_df = mad_df.loc[medians_df.index]
        return medians_df, mad_df

    def median_by_section(self, remove_outliers, scale):
        """
        Wrapper of (calculate_median_by_section).Calculate gene expression
        median and median absolute deviation (MAD) values for each section.

        Parameters
        ----------
        remove_outliers : bool
            if true remove samples outside range 'median+/-2.5*MAD' before
            calculation of median by section

        scale : bool
            if true median and mad are returned as relative quantities
            (normalized by row sum)

        Returns
        -------
        medians_df : DataFrame
            DataFrame with median values per section. If the number of zero
            measurment is above 'sample_t_0' DataFrame will be empty.

        mad_df : DataFrame
            DataFrame with MAD for eah section.
        """
        load_results_median = self.check_step_completion('/'.join([self.sample_by_section, 'median_by_sections.csv']))
        load_results_mad = self.check_step_completion('/'.join([self.sample_by_section, 'mad_by_sections.csv']))
        if load_results_median.empty or load_results_mad.empty:
            medians = pd.DataFrame()
            mad = pd.DataFrame()
            if self.data_type == 'numeric':
                calc_res = Parallel(n_jobs=self.cores)(delayed(self.calculate_median_by_section_numeric)(group, remove_outliers) for i, group in self.data_table.groupby(np.arange(len(self.data_table)) // self.cores))
            elif self.data_type == 'binary':
                calc_res = Parallel(n_jobs=self.cores)(delayed(self.calculate_median_by_section_binary)(group) for i, group in self.data_table.groupby(np.arange(len(self.data_table)) // self.cores))

            for res in calc_res:
                if not res[0].empty:
                    medians = medians.append(res[0])
                    mad = mad.append(res[1])

            if scale:
                for index, row in medians.iterrows():
                    medians.loc[index, :] = row/row.sum()
                for index, row in mad.iterrows():
                    mad.loc[index, :] = row/row.sum()
            medians.to_csv('/'.join([self.sample_by_section, 'median_by_sections.csv']))
            mad.to_csv('/'.join([self.sample_by_section, 'mad_by_sections.csv']))
        else:
            medians = load_results_median
            medians.columns.values[0] = self.index_col
            medians.set_index(self.index_col, inplace=True)
            mad = load_results_mad

        return medians, mad

    def polynomial_fitting(self, table, i, mad):
        """POLYNOMIAL FITTING FOR EACH ROW OF TABLE."""
        poly_fit_results = pd.DataFrame(index=table.index)
        fitting_score = pd.DataFrame(columns=['feature', i])
        models = {}
        for index, row in table.iterrows():
            model = np.polyfit(self.x, row.to_list(), i)
            model = model[::-1]
            temp = pd.Series([index, r2_score(row.to_list(),
                             np.polynomial.polynomial.polyval(self.x, model))],
                             index=fitting_score.columns)
            fitting_score = fitting_score.append(temp, ignore_index=True)
            models[index] = model
        fitting_score.set_index('feature', inplace=True)
        poly_fit_results = poly_fit_results.merge(fitting_score,
                                                  left_index=True,
                                                  right_index=True)
        return poly_fit_results, models

    def sigmoid_func(self, x, x0, y0, c, k):
        return c / (1 + np.exp(-k*(x-x0))) + y0

    def sigmoid_fitting(self, table, guess_bounds, mad, dog_allowed):
        """SIGMOIDAL FITTING FOR EACH ROW OF TABLE."""
        fitting_score = pd.DataFrame(columns=['feature', 'sigmoidal'])
        models = {}
        for index, row in table.iterrows():
            y0_min = row.min()
            y0_max = row.max()
            self.bounds = ([0, -y0_max, -y0_max, -1000], [9, y0_max, y0_max, 1000])
            try:
                if guess_bounds and dog_allowed:
                    parameters, pcov = curve_fit(self.sigmoid_func,
                                                 self.x,
                                                 row.to_list(),
                                                 method='dogbox',
                                                 maxfev=3000,
                                                 bounds=self.bounds)
                elif dog_allowed is False and guess_bounds is False:
                    parameters, pcov = curve_fit(self.sigmoid_func,
                                                 self.x,
                                                 row.to_list(),
                                                 maxfev=3000)
                elif dog_allowed is False:
                    parameters, pcov = curve_fit(self.sigmoid_func,
                                                 self.x,
                                                 row.to_list(),
                                                 bounds=self.bounds,
                                                 maxfev=3000)
                else:
                    parameters, pcov = curve_fit(self.sigmoid_func,
                                                 self.x,
                                                 row.to_list(),
                                                 method='dogbox',
                                                 maxfev=3000)
                score = r2_score(row.to_list(), self.sigmoid_func(self.x, parameters[0], parameters[1],
                                                      parameters[2], parameters[3]))
            except (RuntimeError) as e:
                print('no solution was found!')
                score = np.nan
                parameters = [-999, -999, -999, -999]

            temp = pd.Series([index, score], index=fitting_score.columns)
            fitting_score = fitting_score.append(temp, ignore_index=True)
            models[index] = parameters
        fitting_score.set_index('feature', inplace=True)
        return fitting_score, models

    def fit_data(self, table, mad, guess_bounds=True, dog_allowed=True):
        """Fit continuum and sigmoid models on median data.

        Parameters
        ----------
        table : DataFrame
            median by sections table

        Returns
        -------
        models_scores : DataFrame
            scores of all the models tested
        poly_scores : DataFrame
            scores of polynomial models
        sigmoid_scores : DataFrame
            scores of sigmoidal models
        poly_models : Dictionary
            fitted polynomial
        sig_models : Dictionary
            fitted sigmoid
        """
        load_results1 = self.check_step_completion('/'.join([self.data_fitting, 'polynomial_scores.pkl']), pkl=1)
        load_results2 = self.check_step_completion('/'.join([self.data_fitting, 'sigmoid_scores.pkl']), pkl=1)
        load_results3 = self.check_step_completion('/'.join([self.data_fitting, 'polynomial_models.pkl']), pkl=1)
        load_results4 = self.check_step_completion('/'.join([self.data_fitting, 'sigmoid_models.pkl']), pkl=1)
        if load_results1.empty or load_results2.empty:
            poly_scores = pd.DataFrame(index=table.index)
            sigmoid_scores = pd.DataFrame()
            poly_models = {}
            sig_models = {}

            results = Parallel(n_jobs=self.cores)(delayed(self.polynomial_fitting)(table, degree, mad) for degree in range(1, self.degree_2_test+1))
            for result in results:
                poly_scores = pd.concat([poly_scores, result[0]], axis=1)
                poly_models[result[0].columns[0]] = result[1]
            results = Parallel(n_jobs=self.cores)(delayed(self.sigmoid_fitting)(group, guess_bounds, mad, dog_allowed) for i, group in table.groupby(np.arange(len(table)) // self.cores))
            for result in results:
                sigmoid_scores = sigmoid_scores.append(result[0])
                sig_models.update(result[1])
            poly_scores.to_csv('/'.join([self.data_fitting, 'polynomial_scores.csv']))
            sigmoid_scores.to_csv('/'.join([self.data_fitting, 'sigmoid_scores.csv']))

            open_file = open('/'.join([self.data_fitting, 'polynomial_scores.pkl']), "wb")
            pickle.dump(poly_scores, open_file)
            open_file.close()
            open_file = open('/'.join([self.data_fitting, 'sigmoid_scores.pkl']), "wb")
            pickle.dump(sigmoid_scores, open_file)
            open_file.close()
            open_file = open('/'.join([self.data_fitting, 'sigmoid_models.pkl']), "wb")
            pickle.dump(sig_models, open_file)
            open_file.close()
            open_file = open('/'.join([self.data_fitting, 'polynomial_models.pkl']), "wb")
            pickle.dump(poly_models, open_file)
            open_file.close()
        else:
            poly_scores = load_results1
            sigmoid_scores = load_results2
            poly_models = load_results3
            sig_models = load_results4
            sigmoid_scores.columns.values[0] = 'sigmoidal'
        models_scores = poly_scores
        models_scores['sigmoidal'] = sigmoid_scores['sigmoidal']
        return models_scores, poly_scores, sigmoid_scores, poly_models, sig_models

    def random_polynomial_fitting(self, table, mad):
        """POLYNOMIAL FITTING with RANDOM PERMUTATION."""
        poly_rnd_scores = []
        for degree in range(1, self.degree_2_test+1):
            print(f'Fitting random permutated ({self.rnd_perm_n} times) data with polynomial of degree: {degree}')
            poly_fit_perm_score = pd.DataFrame(index=table.index)
            for i in range(0, self.rnd_perm_n):
                print(f'permutation number {i} of {self.rnd_perm_n}')
                results = self.polynomial_fitting(table.sample(frac=1, axis=1), degree, mad.sample(frac=1, axis=1))
                poly_fit_perm_score = poly_fit_perm_score.merge(results[0], left_index=True, right_index=True, suffixes=(f'_{i-1}', f'_{i}'))
            poly_rnd_scores.append(poly_fit_perm_score)
        return poly_rnd_scores

    def random_sigmoidal_fitting(self, table, guess_bounds, mad, dog_allowed):
        """SIGMOIDAL FITTING with RANDOM PERMUTATION."""
        random_sig_df = pd.DataFrame(index=table.index)
        sig_rand_param = {}
        print(f'Fitting random permutated ({self.rnd_perm_n} times) data with simoid model')
        for i in range(0, self.rnd_perm_n):
            results = self.sigmoid_fitting(table.sample(frac=1, axis=1), guess_bounds, mad.sample(frac=1, axis=1), dog_allowed)
            sig_rand_param[f'sig_p_{i}'] = results[1]
            random_sig_df = random_sig_df.merge(results[0], left_index=True, right_index=True, suffixes=(f'_{i-1}', f'_{i}'))
        return random_sig_df

    def fit_random_data(self, table, mad, guess_bounds=False, dog_allowed=True):
        """Fit continuum and sigmoid models on random permutation of
        median data.
        Parameters
        ----------
        table : DataFrame
            median by sections table

        Returns
        -------

        """
        load_results1 = self.check_step_completion('/'.join([self.rnd_data_fitting, 'polynomial_random_fitting.pkl']), pkl=1)
        load_results2 = self.check_step_completion('/'.join([self.rnd_data_fitting, 'sigmoidal_random_fitting.pkl']), pkl=1)
        if load_results2.empty:
            poly_random_result = Parallel(n_jobs=self.cores)(delayed(self.random_polynomial_fitting)(group, mad) for i, group in tqdm(table.groupby(np.arange(len(table)) // 10)))
            print('done polynomial')
            sigmoid_random_result = Parallel(n_jobs=self.cores)(delayed(self.random_sigmoidal_fitting)(group, guess_bounds, mad, dog_allowed) for i, group in tqdm(table.groupby(np.arange(len(table)) // 10)))
            rand_fitting_score_poly = []
            for i in range(0, self.degree_2_test):
                degree_results = pd.DataFrame()
                for result in poly_random_result:
                    degree_results = degree_results.append(result[i])
                rand_fitting_score_poly.append(degree_results)
            rand_fitting_score_sig = pd.DataFrame()
            for result in sigmoid_random_result:
                rand_fitting_score_sig = rand_fitting_score_sig.append(result)

            open_file = open('/'.join([self.rnd_data_fitting, 'polynomial_random_fitting.pkl']), "wb")
            pickle.dump(rand_fitting_score_poly, open_file)
            open_file.close()
            open_file = open('/'.join([self.rnd_data_fitting, 'sigmoidal_random_fitting.pkl']), "wb")
            pickle.dump(rand_fitting_score_sig, open_file)
            open_file.close()
        else:
            rand_fitting_score_poly = load_results1
            rand_fitting_score_sig = load_results2
        return rand_fitting_score_poly, rand_fitting_score_sig

    def gof_dist(self, obs_score_list, perm_score_list, degree, dist_obs, dist_perm, pdf_perm, pdf_obs, vline):
        a, b, loc, scale = beta.fit(perm_score_list)
        a1, b1, loc1, scale1 = beta.fit(obs_score_list)

        x = np.linspace(0.01, 1, 1000)
        pdf = beta.pdf(x, a, b, loc, scale)

        pdf_data = beta.pdf(x, a1, b1, loc1, scale1)

        threshold = beta.ppf(self.t_area, a, b, loc, scale)
        bins = np.histogram(np.hstack((obs_score_list, perm_score_list)), bins=100)[1]
        plt.figure(figsize=(10, 10))
        # plt.title(f'model degree: {degree}')
        if dist_obs:
            graph = plt.hist(obs_score_list, bins=bins, density=True, color='green', edgecolor='k', linewidth=1.2, alpha=0.5, label='observations');
        if dist_perm:
            graph = plt.hist(perm_score_list, bins=bins, density=True, color='red', edgecolor='k', linewidth=1.2, alpha=0.5, label='random permutations');
        if pdf_perm:
            plt.plot(x, pdf, 'r', linewidth=3, label='Permutations')
            max_pdf = max(pdf)
        if pdf_obs:
            plt.plot(x, pdf_data, 'g', linewidth=3, label='Observation')
            if max_pdf < max(pdf_data):
                max_pdf = max(pdf_data)
        if 'graph' in locals():
            ylim = ceil(max(graph[0])+max(graph[0])*0.1)
        else:
            ylim = max_pdf
        plt.ylim(0, ylim)
        if vline:
            plt.vlines(beta.ppf(self.t_area, a, b, loc, scale), 0, ylim, linestyles='--', color='k', label=f'{self.t_area*100}% threshold')

        threshold = beta.ppf(self.t_area, a, b, loc, scale)
        t = np.arange(threshold, 1, 0.01)
        plt.fill_between(x=t, y1=beta.pdf(t, a1, b1, loc1, scale1),
            where= (-1 < t)&(t < 1),
            color= "g",
            alpha= 0.2)
        plt.legend(loc='upper left')
        plt.ylabel('Density')
        plt.xlabel('R score')
        plt.tight_layout()
        plt.savefig('/'.join([self.figures, f'{degree}.svg']), format="svg")
        plt.show()
        return threshold

    def gof_performance(self, poly_obs_score_lists, sig_obs_score_lists, poly_perm_score_lists, sig_perm_score_lists):
        plot_list = []
        c = ['green', 'red','green', 'red','green', 'red','green', 'red','green', 'red']
        medianprops = dict(color = 'k', linestyle='-')
        boxprops = dict(linestyle='-', linewidth=1.5, color='k')
        for i in range(0, self.degree_2_test):
            plot_list.append(poly_obs_score_lists[i])
            plot_list.append(poly_perm_score_lists[i])
        plot_list.append(sig_obs_score_lists)
        plot_list.append(sig_perm_score_lists)
        plt.figure(figsize=(15, 10))
        bplot = plt.boxplot(plot_list, showfliers=False, patch_artist=True, medianprops=medianprops, boxprops=boxprops)
        for patch, color in zip(bplot['boxes'], c):
            patch.set_alpha(0.6)
            patch.set_facecolor(color)
        labels = []
        for i in range(1, self.degree_2_test+1):
            labels.append(str(i))
            labels.append(f'{i}_{self.rnd_perm_n}_permutations')
        labels.append('sigmoid')
        labels.append(f'sigmoid_{self.rnd_perm_n}_permutations')
        plt.xticks(range(1, 2*self.degree_2_test+3), labels, rotation=45, ha='right')
        plt.ylabel('R2')
        plt.xlabel('Degree')
        plt.tight_layout()
        plt.savefig('/'.join([self.figures, 'gof_normal_vs_perm.svg']), format='svg', transparent=True)
        plt.show()

    def plot_gof(self, poly_obs_fit_scores, sig_obs_fit_scores, poly_perm_fit_scores, sig_perm_fit_scores, dist_obs=True, dist_perm=True, pdf_perm=True, pdf_obs=True, vline=True):
        """
        Plot r score distribution for normal and permutated data for each model analysed
        """
        poly_perm_score_lists = []
        for i in range(0, self.degree_2_test):
            temp = []
            for column in poly_perm_fit_scores[i].columns:
                temp = temp+poly_perm_fit_scores[i][column].to_list()
            poly_perm_score_lists.append(temp)
        sig_perm_score_lists = []
        temp = []
        for column in sig_perm_fit_scores.columns:
            temp = temp+sig_perm_fit_scores[sig_perm_fit_scores[column]>0.01][column].to_list()
        sig_perm_score_lists.append(temp)

        sig_obs_score_lists = sig_obs_fit_scores['sigmoidal'].dropna().tolist()
        sig_obs_score_lists = [item for item in sig_obs_score_lists if item >= 0]
        poly_obs_score_lists = []
        for column in poly_obs_fit_scores.columns:
            poly_obs_score_lists.append(poly_obs_fit_scores[column].to_list())

        self.gof_performance(poly_obs_score_lists, sig_obs_score_lists, poly_perm_score_lists, sig_perm_score_lists)

        sig_threshold = self.gof_dist(sig_obs_score_lists, sig_perm_score_lists[0], 'sigmoid', dist_obs, dist_perm, pdf_perm, pdf_obs, vline)
        # stats, pvalue = mannwhitneyu(sig_obs_score_lists, sig_perm_score_lists[0])
        stats, pvalue = mannwhitneyu(sig_obs_score_lists, sig_perm_score_lists[0], alternative='greater')
        stats_models = {}
        stats_models['sigmoidal'] = [stats, pvalue, sig_threshold]
        for i in range(0, len(poly_perm_score_lists)):
            poly_threshold = self.gof_dist(poly_obs_score_lists[i], poly_perm_score_lists[i], i+1, dist_obs, dist_perm, pdf_perm, pdf_obs, vline)
            # stats, pvalue = mannwhitneyu(poly_obs_score_lists[i], poly_perm_score_lists[i])
            stats, pvalue = mannwhitneyu(poly_obs_score_lists[i], poly_perm_score_lists[i], alternative='greater')
            stats_models[i+1] = [stats, pvalue, poly_threshold]

        self.stats_models = stats_models
        pd.DataFrame(stats_models).to_csv('/'.join([self.output, 'stats_models.csv']))
        return stats_models

    def cluster_genes(self, models_scores):
        genes_clusters = {}
        for column in models_scores.columns:
            temp = models_scores.sort_values(column, ascending=False)
            temp = temp[temp[column] > self.stats_models[column][2]]
            genes_clusters[column] = set(temp.index)
        return genes_clusters

    def plot_clusters(self, genes_clusters):
        clusters = []
        labels = []
        for key in genes_clusters:
            clusters.append(genes_clusters[key])
            labels.append(key)
        plt.figure(figsize=(20, 10))
        supervenn(clusters, labels)
        plt.title("Overlap between models")
        plt.tight_layout()
        plt.savefig('/'.join([self.figures, 'super_venn.svg']) format='svg')
        plt.show()

    def get_summary_table(self, genes_clusters, models_scores):
        for key in self.stats_models.keys():
            if self.stats_models[key][1] > 0.05 and key in genes_clusters:
                genes_clusters.pop(key)
        clustered_genes_list = []
        for key in genes_clusters.keys():
            clustered_genes_list = clustered_genes_list+list(genes_clusters[key])
        clustered_genes_list = set(clustered_genes_list)
        summary_table = pd.DataFrame()
        for key in genes_clusters.keys():
            for gene in clustered_genes_list:
                if gene in genes_clusters[key]:
                    summary_table.loc[gene, key] = True
                    summary_table.loc[gene, f'{key}_score'] = models_scores.loc[gene, key]
                else:
                    summary_table.loc[gene, key] = False
                    summary_table.loc[gene, f'{key}_score'] = 0
            summary_table[key] = summary_table[key].astype('bool')
        summary_table.to_csv('/'.join([self.output, 'summary.csv']))
        return summary_table

    def classify_genes(self, summary):
        if 'sigmoidal' in summary.columns:
            cont = summary[summary['sigmoidal'] == False]
            sig_raw = summary[summary['sigmoidal'] == True]
            discarded = pd.DataFrame(columns=sig_raw.columns)
            sigmoid = pd.DataFrame(columns=sig_raw.columns)
            cont_significant_status = sig_raw.select_dtypes(include='bool')
            cont_score = sig_raw.select_dtypes(include='float')
            cont_score.columns = cont_significant_status.columns
            cont_score.drop(['sigmoidal'], axis=1, inplace=True)
            cont_significant_status.drop(['sigmoidal'], axis=1, inplace=True)

            for index, row in sig_raw.iterrows():
                if (cont_significant_status.loc[index] == True).any():
                    indexes_to_compare = list(cont_significant_status.loc[index][cont_significant_status.loc[index]==True].index)
                    max_index = cont_score.loc[index, indexes_to_compare].idxmax()
                    discard = 2

                    if row[f'{max_index}_score']-row['sigmoidal_score'] > 0.2:
                        discard = 1
                    elif row['sigmoidal_score'] - row[f'{max_index}_score'] > 0.2:
                        discard = 0

                    if discard == 1:
                        cont.loc[index] = row
                    elif discard == 0:
                        sigmoid.loc[index] = row
                    else:
                        discarded.loc[index] = row
                        print(f'Sigmoid and continuos score for gene {index} are too close, gene will be discarded.')
                else:
                    sigmoid.loc[index] = row
        else:
            cont = summary
            sigmoid = pd.DataFrame()
            discarded = pd.DataFrame()

        cont = cont.select_dtypes(include='float')
        sigmoid = sigmoid.select_dtypes(include='float')
        discarded = discarded.select_dtypes(include='float')
        continuos_res = pd.DataFrame()
        sigmoid_res = pd.DataFrame()
        discarded_res = pd.DataFrame()
        for index, row in cont.iterrows():
            continuos_res.loc[index, 'model'] = cont.loc[index].idxmax().split('_')[0]
            continuos_res.loc[index, 'score'] = cont.loc[index, cont.loc[index].idxmax()]
        for index, row in sigmoid.iterrows():
            sigmoid_res.loc[index, 'model'] = 'sigmoid'
            sigmoid_res.loc[index, 'score'] = sigmoid.loc[index, 'sigmoidal_score']
        for index, row in discarded.iterrows():
            discarded_res.loc[index, 'model'] = 'sigmoid'
            # discarded_res.loc[index, 'score'] = sigmoid.loc[index, 'sigmoidal_score']
        sigmoid_res.to_csv('/'.join([self.output, 'sigmoidal.csv']))
        continuos_res.to_csv('/'.join([self.output, 'continuum.csv']))
        discarded_res.to_csv('/'.join([self.output, 'discarded.csv']))
        return continuos_res, sigmoid_res, discarded_res

    def plot_fitting(self, scores_table, gene_indexes_list, medians, poly_models, sig_models, model, boxplots=True, save_as='', plot_fit=True, ylabel='', title=True, set_lim=[]):
        medianprops = dict(linestyle='None')
        boxprops = dict(linestyle='-', linewidth=1.5, color='k')
        if (len(gene_indexes_list) % 2 != 0):
            vertical = len(gene_indexes_list)//2+1
        else:
            vertical = len(gene_indexes_list)//2
        h = 0
        l = 0
        plt.figure(figsize=(20, 10*vertical))
        for key in gene_indexes_list:
            x = np.linspace(self.x[0], self.x[-1], 1000)
            axs = plt.subplot2grid((vertical, 2), (h, l))
            plt.plot(self.x, list(medians.loc[key,:]), color='r', marker='o', ls='', markeredgecolor='k', markeredgewidth=2, markersize=10)
            gene_dist = []
            if boxplots:
                for section in medians.columns:
                    gene_dist.append(self.data_table.loc[key, self.samples2sections[section]].to_list())
                gene_dist = np.array(gene_dist, dtype=object)
                plt.boxplot(gene_dist, showfliers=False, medianprops=medianprops, boxprops=boxprops)
            if model == 'sigmoid':
                degree = 'sigmoid'
                y = self.sigmoid_func(x, sig_models[key][0], sig_models[key][1], sig_models[key][2], sig_models[key][3])
            elif model == 'continuum':
                degree = int(scores_table.loc[key,'model'])
                y = np.polynomial.polynomial.polyval(x, poly_models[degree][key])
            elif model == 'both':
                degree = int(scores_table.loc[key,'model'])
                y1 = self.sigmoid_func(x, sig_models[key][0], sig_models[key][1], sig_models[key][2], sig_models[key][3])
                y2 = np.polynomial.polynomial.polyval(x, poly_models[degree][key])
            if key in scores_table.index:
                score = round(scores_table.loc[key,'score'], 3)
            else:
                score = 'NA'
            if title:
                plt.title(f'{key}, model:{degree}, score:{score}')
            if plot_fit and model!='both':
                plt.plot(x, y, color='red', ls='-', linewidth=2)
            elif plot_fit and model=='both':
                plt.plot(x, y1, color='red', ls='-', linewidth=2)
                plt.plot(x, y2, color='green', ls='-', linewidth=2)
            if set_lim:
                plt.ylim(set_lim)
            plt.xticks(ticks=self.x, labels=self.sections4plots, rotation=45, ha='right', fontsize=22)
            plt.ylabel(ylabel, fontsize=22)
            l = l+1
            if l == 2:
                h = h+1
                l = 0
        plt.tight_layout()
        if save_as:
            plt.savefig('/'.join([self.figures, f'{save_as}']), format="svg")
        plt.show()

    def plot_fitting_bars(self, scores_table, gene_indexes_list, medians, mad, poly_models, sig_models, model, save_as='', plot_fit=True, ylabel='', title=True, set_lim=[], plot_mad=True):
        medianprops = dict(linestyle='None')
        boxprops = dict(linestyle='-', linewidth=1.5, color='k')
        if (len(gene_indexes_list) % 2 != 0):
            vertical = len(gene_indexes_list)//2+1
        else:
            vertical = len(gene_indexes_list)//2
        h = 0
        l = 0
        plt.figure(figsize=(20, 10*vertical))
        for key in gene_indexes_list:
            x = np.linspace(self.x[0], self.x[-1], 1000)
            axs = plt.subplot2grid((vertical, 2), (h, l))
            if plot_mad:
                plt.bar(self.x, list(medians.loc[key,:]),yerr=list(mad.loc[key,:]), color='grey', edgecolor='k')
            else:
                plt.bar(self.x, list(medians.loc[key,:]), color='grey', edgecolor='k')
            gene_dist = []
            if model == 'sigmoid':
                degree = 'sigmoid'
                y = self.sigmoid_func(x, sig_models[key][0], sig_models[key][1], sig_models[key][2], sig_models[key][3])
            elif model == 'continuum':
                degree = int(scores_table.loc[key,'model'])
                y = np.polynomial.polynomial.polyval(x, poly_models[degree][key])
            elif model == 'both':
                degree = int(scores_table.loc[key,'model'])
                y1 = self.sigmoid_func(x, sig_models[key][0], sig_models[key][1], sig_models[key][2], sig_models[key][3])
                y2 = np.polynomial.polynomial.polyval(x, poly_models[degree][key])
            if key in scores_table.index:
                score = round(scores_table.loc[key,'score'], 3)
            else:
                score = 'NA'
            if title:
                plt.title(f'{key}, model:{degree}, score:{score}')
            if plot_fit and model!='both':
                plt.plot(x, y, color='red', ls='-', linewidth=2)
            elif plot_fit and model=='both':
                plt.plot(x, y1, color='red', ls='-', linewidth=2)
                plt.plot(x, y2, color='green', ls='-', linewidth=2)
            if set_lim:
                plt.ylim(set_lim)
            plt.xticks(ticks=self.x, labels=self.sections4plots, rotation=45, ha='right', fontsize=22)
            plt.ylabel(ylabel, fontsize=22)
            l = l+1
            if l == 2:
                h = h+1
                l = 0
        plt.tight_layout()
        if save_as:
            plt.savefig('/'.join([self.figures, f'{save_as}']), format="svg")
        plt.show()

    def copytree(self, src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    def save_project(self, save_to, overwrite=False):
        fromDirectory = self.project_path
        toDirectory = save_to
        if os.path.isdir(toDirectory) and overwrite:
            self.copytree(fromDirectory, toDirectory)
        elif os.path.isdir(toDirectory) and not overwrite:
            print(f"Can't save project, directory already exist and overwrite = {overwrite}")
        elif not os.path.isdir(toDirectory):
            os.mkdir(toDirectory)
            self.copytree(fromDirectory, toDirectory)

    def plot_sample_distribution(self, save_as='samples_distribution.svg'):
        """
        Plot samples distribution in colon sections and save the figure.
        Needs to be executed after create_samples_to_sections_table() or
        an error will be triggered.

        Parameters
        ----------
        save_as : string
            Save figure with the specified name (default: samples_distribution.svg)
        """
        sections = [val for sublist in self.sections.values() for val in sublist]
        samplexsec = {}
        for section in sections:
            group = self.clinical_data[self.clinical_data['site_of_resection_or_biopsy'] == section]
            i = 0
            for index, row in group.iterrows():
                if row['sample_submitter_id'] in self.data_table.columns:
                    i = i+1
            samplexsec[section] = i
        plt.figure(figsize=(10, 8))
        plt.bar(*zip(*samplexsec.items()), edgecolor='k')
        plt.ylabel('Frequency')
        plt.xticks(range(len(samplexsec.keys())), samplexsec.keys(), rotation=45, ha='right')
        plt.tight_layout()
        plt.title('Samples Distribution')
        plt.savefig('/'.join([self.figures, f'{save_as}']), format="svg")

    def strict_sig_list(self, sigmoid_genes, sig_models, plot_dist = False):
        section_l = []
        gene_list = []
        x = sym.Symbol('x')
        x0 = sym.Symbol('x0')
        y0 = sym.Symbol('y0')
        c = sym.Symbol('c')
        k = sym.Symbol('k')
        f = c / (1 + sym.exp(-k*(x-x0))) + y0
        f_prime = f.diff(x)
        f_prime = sym.lambdify([(x, x0, y0, c, k)], f_prime)
        x = np.linspace(self.x[0], self.x[-1], 2000)
        plt.figure(figsize=(10, 10))
        for key in list(sigmoid_genes.index):
            d_ = []
            for i in x:
                d_.append(abs(f_prime(np.insert(sig_models[key],0,i))))
            index_min = np.argmax(d_)
            section = list(self.sections.keys())[int(round(x[index_min]))-1]
            if section in ['Transverse colon', 'Descending colon']:
                gene_list.append(key)
            section_l.append(section)
        if plot_dist:
            indexes = np.arange(len(self.sections))
    #         width = 0.3
            values = pd.Series(section_l).value_counts().reindex(self.sections)/len(section_l)*100
            plt.bar(indexes, values, edgecolor='k', color = 'grey', linewidth=1.5)
            plt.xticks(indexes, self.sections, rotation=45, ha='right')
            plt.ylabel('Relative Frequency (%)')
            # plt.ylim([0,50])
            plt.title('Distribution of inflexion points')
            plt.tight_layout()
            plt.savefig('/'.join([self.figures, 'inflection_distribution_sigmoid.svg']), format='svg')
        with open('/'.join([self.output, 'strict_sig_genes_list.txt']), 'w') as f:
            for item in gene_list:
                f.write("%s\n" % item)
        return gene_list, section_l

    def strict_sig_list_random(self, medians, mad, sigmoid_genes, sig_models, plot_dist=False):
        dog_allowed = False
        input_medians = medians.loc[sigmoid_genes.index]
        j = 0
        randomized_data = pd.DataFrame(columns=range(0, len(medians.columns)))
        for i in range(0, 10):
            for index, row in input_medians.iterrows():
                randomized_data.loc[j] = row.sample(frac=1).values
                j=j+1
        guess_bounds = True
        results = Parallel(n_jobs=self.cores)(delayed(self.sigmoid_fitting)(group, guess_bounds, mad, dog_allowed) for i, group in randomized_data.groupby(np.arange(len(randomized_data)) // self.cores))
        sigmoid_genes_rnd = results[0][0]
        sig_models_rnd = results[0][1]
        section_l = []
        gene_list = []
        x = sym.Symbol('x')
        x0 = sym.Symbol('x0')
        y0 = sym.Symbol('y0')
        c = sym.Symbol('c')
        k = sym.Symbol('k')
        f = c / (1 + sym.exp(-k*(x-x0))) + y0
        f_prime = f.diff(x)
        f_prime = sym.lambdify([(x, x0, y0, c, k)], f_prime)
        x = np.linspace(self.x[0], self.x[-1], 2000)
        for key in list(sigmoid_genes_rnd.index):
            d_ = []
            for i in x:
                d_.append(abs(f_prime(np.insert(sig_models_rnd[key], 0, i))))
            index_min = np.argmax(d_)
            section = list(self.sections.keys())[int(round(x[index_min]))-1]
            if section in ['Transverse colon', 'Descending colon']:
                gene_list.append(key)
            section_l.append(section)
        indexes = np.arange(len(self.sections))
        #         width = 0.3
        values = pd.Series(section_l).value_counts().reindex(self.sections)/len(section_l)*100
        plt.figure(figsize=(10, 10))
        plt.bar(indexes, values, edgecolor='k', color='grey', linewidth=1.5)
        plt.xticks(indexes, self.sections, rotation=45, ha='right')
        plt.ylabel('Relative Frequency (%)')
        # plt.ylim([0,50])
        plt.title('Distribution of inflexion points')
        plt.tight_layout()
        plt.savefig('/'.join([self.figures, 'inflection_distribution_sigmoid_random.svg']), format='svg')
