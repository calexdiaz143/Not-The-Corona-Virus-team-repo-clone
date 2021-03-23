#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
import os


class Grader(object):
    def __init__(self):
        self.columns = ['ID', 'Health', 'Rate of health decline', 'Bed', 'Ventilator', 'Supplemental oxygen', 'Remdesivir',
                        'Dexamethasone', 'Plasma', 'Casirivimab', 'Chloroquine', 'Total']
        self.treatments = [('Bed', 30, 5), ('Ventilator', 10, 30), ('Supplemental oxygen', 10, 20),
                           ('Remdesivir', 7, 30), ('Dexamethasone', 20, 25), ('Plasma', 10, 15),
                           ('Casirivimab', 10, 15), ('Chloroquine', 17, 10)]
        
    #public
    
    def validate(self, subm, data):
        if type(subm) is pd.DataFrame:
            return self.validate_one(subm, data)
        if type(subm) is list:
            return self.validate_multiple(subm, data)
        else:
            print("Submission and data types must match and be either dataframe or list of dataframes")
    
    def validate_one(self, df, data):
        return self._validate(df, data)
    
    def validate_multiple(self, subm_lst, data_lst):
        score = 0
        assert subm_lst, "No valid submission file found.  Submissions must be named: Challenge3_Day<day number>Submission.csv"
        #while subm_lst[-1] is None: del subm_lst[-1]  # trim None from end of list
        results = []
        for i,(subm,data) in enumerate(zip(subm_lst, data_lst)):
            
            try:
                s = self._validate(subm, data, i+1, show=False)
                results.append(s)
            except AssertionError as e:
                s = 0
                print(f"Invalid entry on day {i+1}: {e}") 
                results.append(e)
                continue
                
            score += s #self._validate(subm, data, i+1, show=False)
        print(f"Total score for {len(subm_lst)} days: {score}", end='\n\n')
        assert type(results[-1]) is np.int64, results[-1]   
        return score
            
    
    #private   
    
    def _validate(self, df, data, day_num=None, show=True):
        self._convert_from_binary(df)
        self._verify_column_names(df)
        self._verify_data_match(df, data)
        self._verify_treatments(df)
        self._subtract_treatments_used(df)
        self._verify_treatments_with_bed(df)
        self._verify_combo_constraints(df)
        return self._scorer(df, day_num, show)   
    
    def _convert_from_binary(self, df):
        for name, qty, eff in self.treatments:
            if name in df:
                df[name][df[name] > 0] = eff        
        
    def _verify_column_names(self, df):
        '''verify the csv is well-formatted and treatments are spelled correctly'''
        for c in df.columns:
            assert c in self.columns, f"Column: {c} is not an accepted column name"

    def _verify_data_match(self, df, data):
        '''verify the IDs, Health, Rate of health decline matches the day's data file'''
        dd = pd.merge(data, df, on=['ID','Health','Rate of health decline'], how='left', indicator='exist')        
        assert np.all(dd.exist=='both'), "submission does not match data on ID, Health, or Rate of health decline"

    def _verify_treatments(self, df):
        '''verify that treatment counts and efficacies are correct'''
        for name, qty, eff in self.treatments:
            if name in df.columns: 
                m = df[name]==eff
                num_used = (m).sum()
                assert num_used <= qty, f"exceeded treatment quantity for {name}"
                assert np.all(df[name][~m]==0) , f"incorrect treatment efficacy used for {name}"
                
    def _subtract_treatments_used(self, df):
        for t_idx in range(3,len(self.treatments)):  # subtract only from one-time treatments
            name, qty, eff = self.treatments[t_idx]
            if name in df.columns:
                m = df[name]==eff
                num_used = (m).sum()
                self.treatments[t_idx] = (name, qty-num_used, eff)
    
    def _verify_treatments_with_bed(self, df):
        '''verify that treatments only occur for sailors with beds'''
        no_beds = df[df['Bed'] == 0]
        for col in self.columns[4:-1]:
            if col in no_beds.columns:
                err = f"applying {col} to a sailor without a bed. "
                err += "Only sailors assigned to beds can receive treatment."
                assert np.all(no_beds[col].values==0), err

    def _verify_combo_constraints(self, df):
        '''verify that oxygen cannot be used with ventilators constraints'''
        vent = df[df['Ventilator'] != 0]
        if 'Supplemental oxygen' in df.columns:
            err = f"cannot combine Supplemental oxygen with Ventilator"
            assert np.all(vent['Supplemental oxygen'].values==0), err
            
    def _calc_total(self, df):
        '''calculate totals to ensure they are correct'''
        return df.drop(['ID', 'Total'], axis=1).sum(axis=1)
        
    def _calc_bonuses_reusable(self, df, show):
        '''calculate bonus for conserved reusable treatments (not including bed)'''
        total = 0
        for name, qty, eff in self.treatments[1:3]:
            try:
                num_used = (df[name] != 0).astype(int).sum()
            except KeyError:
                num_used = 0
            num = qty - num_used
            bonus = int(eff * 0.5 * num)
            if show: print(f"Bonus for {name} ({num} remaining): {bonus}")
            total += bonus
        return total
    
    def _calc_bonuses_onetime(self, df, show):
        '''calculate bonus for conserved treatments (not including bed)'''
        total = 0
        for name, qty, eff in self.treatments[3:]:
            bonus = int(eff * 0.5 * qty)
            if show: print(f"Bonus for {name} ({qty} remaining): {bonus}")
            total += bonus
        return total

    def _calc_penalty(self, totals, show):
        '''calculate penalty for a sailor dying'''
        dead = (totals<=0).sum()
        penalty = dead * 100
        if show: print(f"{dead} sailors died.  Penalty: -{penalty}")
        return penalty

    def _scorer(self, df, day_num, show):
        '''calculate and print total score for a given day'''
        total = self._calc_total(df)
        score = np.clip(total, 0, 100).sum()  # clip Health btw 0 and 100
        if show: print(f"Health total: {score}")
        score -= self._calc_penalty(total, show)
        score += self._calc_bonuses_reusable(df, show)
        score += self._calc_bonuses_onetime(df, show)
        print(f"Total score for day{' '+str(day_num) if day_num else ''}: {score}")
        return score



def read_data_csv(data_path):
    return pd.read_csv(data_path, header=1, usecols=[0,1,2])


if __name__=="__main__":
    subm_path = './Challenge3/submissions/'
    subm = [pd.read_csv(subm_path+d) for d in sorted(os.listdir(subm_path)) if d in ['Challenge3_Day1Submission.csv', 'Challenge3_Day2Submission.csv', 'Challenge3_Day3Submission.csv']]
    data_path = './Challenge3/data/'
    data = [read_data_csv(data_path+d) for d in sorted(os.listdir(data_path)) if d in ['day1data.csv', 'day2data.csv', 'day3data.csv']]
    g = Grader().validate(subm, data)
