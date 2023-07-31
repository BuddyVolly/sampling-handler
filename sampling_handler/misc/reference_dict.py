REFERENCE_DICT = {
    'project_params': {
        'project_name': {'type': 'project_name'},   # we check for spcae in the check routine
        'project_dir': {'type': 'project_dir'},
        'aoi': {'type': 'aoi'},
        'aoi_crs': {'type': 'projection'}
    },
    'stats_params': {
        'start': {"type": int, "choices": range(2001, 2023)},
        'end': {"type": int, "choices": range(2001, 2023)},
        'tree_cover': {"type": int, "choices": range(1, 101)},
        'mmu': {"type": (int, float), "choices": {'min': 0, 'max': 10000}},
        'target_error': {"type": (int, float), "choices":  {'min': 1, 'max': 100}},
        'confidence': {"type": (int, float), "choices":  {'min': 1, 'max': 100}},
        'optimal_spacing': {"type": (int, float), "choices":  {'min': 1, 'max': 100000000}},
        'spacings': {'type': (int, list), 'list_type': int, 'choices': range(1, 1000001)},
        'scales':  {'type': (int, list), 'list_type': int, 'choices': range(1, 1000001)},
        'runs':  {'type': int, 'choices': range(1, 2147483648)},
        'random_seed': {'type': int, 'choices': range(-2147483648, 2147483648)},
        'area_dict':  {
            'total_area':  {'type': (int, float), "choices":  {'min': 1, 'max': 1e16}},
            'forest_area':  {'type': (int, float), "choices":  {'min': 1, 'max': 1e16}},
            'change_area':  {'type': (int, float), "choices":  {'min': 1, 'max': 1e16}}
        },
        'outdir': {'type': str}
    },
    'area_dict': {
        'total_area':  {'type': (int, float), "choices":  {'min': 1, 'max': 1e16}},
        'forest_area':  {'type': (int, float), "choices":  {'min': 1, 'max': 1e16}},
        'change_area':  {'type': (int, float), "choices":  {'min': 1, 'max': 1e16}}
    },
    'design_params': {
        'sampling_strategy': {"type": str, "choices": ['centroid', 'random']},
        'grid_shape': {"type": str, "choices": ['hexagonal', 'squared']},
        'grid_size': {"type": int, "choices": range(30, 100000001)},
        'grid_crs': {"type": 'projection'},
        'out_crs': {"type": 'projection'},
        'dggrid': {
            'projection': {'type': str, "choices": ['ISEAH3']},
            'resolution': {'type': int, 'choices': range(1, 21)},
        },
        'pid': {'type': str},
        'ee_samples_fc': {'type': 'FeatColl'},
        'ee_grid_fc': {'type': 'FeatColl'},
        'outdir': {'type': str}
    },
    'ts_params': {
        'ts_start': {'type': 'date'},
        'ts_end': {'type': 'date'},
        'satellite': {"type": str, "choices": ['Landsat']},
        'lsat_params':  {
            'l9': {'type': bool},
            'l8': {'type': bool},
            'l7': {'type': bool},
            'l5': {'type': bool},
            'l4': {'type': bool},
            'brdf': {'type': bool},
            'bands': {'type': (str, list), 'list_type': str, 'choices': [
                'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndfi', 'ndmi', 'ndvi',
                'brightness', 'greenness', 'wetness'
            ]},
            'max_cc': {'type': int, "choices": range(1, 101)},
        },
        'scale': {'type': int, 'choices': range(1, 100001)},
        'bounds_reduce': {'type': bool},
        'ee_workers': {'type': int, 'choices': range(1, 129)},
        'max_points_per_chunk': {'type': int, 'choices': range(1, 1000001)},
        'grid_size_levels': {
            'type': (int, float, list), 'list_type': float, 'choices': {'min': 0, 'max': 10}
        },
        'outdir': {'type': str}
    },
    'da_params': {
        'start_calibration': {'type': 'date'},
        'start_monitor': {'type': 'date'},
        'end_monitor': {'type': 'date'},
        'ts_band': {'type': str, 'choices': [
                'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndfi', 'ndmi', 'ndvi',
                'brightness', 'greenness', 'wetness'
        ]},
        'outlier_removal': {'type': bool},
        'smooth_ts': {'type': bool},
        'bfast': {
            "run": {'type': bool},
            'start_monitor': {'type': 'date'},
            'freq': {'type': int, 'choices': range(1, 100000)},
            'k': {'type': int, 'choices': range(1, 5)},
            'hfrac': {'type': float, 'choices': {'min': 0.001, 'max': 1}},
            'trend': {'type': bool},
            'level': {'type': float, 'choices': {'min': 0.001, 'max': 1}},
            'backend': {'type': str, 'choices': ['python']},
        },
        'cusum': {
            'run': {'type': bool},
            'nr_of_bootstraps': {'type': int, 'choices': range(1, 100000)},
        },
        'bs_slope': {
            'run': {'type': bool},
            'nr_of_bootstraps': {'type': int, 'choices': range(1, 100000)},
        },
        'ts_metrics': {
            'run': {'type': bool},
            'bands': {'type': (str, list), 'list_type': str, 'choices': [
                'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndfi', 'ndmi', 'ndvi',
                'brightness', 'greenness', 'wetness'
            ]},
            "metrics": {'type': (str, list), 'list_type': str, 'choices': [
                "mean", "stddev", "min", "max"]
            },
            "outlier_removal": {'type': bool},
            "z_threshhold": {'type': int, 'choices': range(1, 100000)},
        },
        'ccdc': {
            'run': {'type': bool},
            'breakpointBands':  {'type': list, 'list_type': str, 'choices': [
                'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndfi', 'ndmi', 'ndvi',
                'brightness', 'greenness', 'wetness'
            ]},
            'tmaskBands': {'type': list, 'list_type': str, 'choices': [
                'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndfi', 'ndmi', 'ndvi',
                'brightness', 'greenness', 'wetness'
            ]},
            'minObservations': {'type': int, 'choices': range(1, 100000)},
            'chiSquareProbability': {'type': float, 'choices': {'min': 0.00001, 'max': 1}},
            'minNumOfYearsScaler':  {'type': int, 'choices': range(1, 100000)},
            'dateFormat':  {'type': int, 'choices': range(1, 4)},
            'lambda':  {'type': int, 'choices': range(0, 100000)},
            'maxIterations': {'type': int, 'choices': range(1, 100000)},
        },
        'land_trendr': {
            'run': {'type': bool},
            'maxSegments': {'type': int, 'choices': range(1, 100000)},
            'spikeThreshold': {'type': float, 'choices': {'min': 0.00001, 'max': 1}},
            'vertexCountOvershoot': {'type': int, 'choices': range(1, 100000)},
            'preventOneYearRecovery':  {'type': bool},
            'recoveryThreshold': {'type': float, 'choices': {'min': 0.00001, 'max': 1}},
            'pvalThreshold': {'type': float, 'choices': {'min': 0.00001, 'max': 1}},
            'bestModelProportion': {'type': float, 'choices': {'min': 0.00001, 'max': 1}},
            'minObservationsNeeded': {'type': int, 'choices': range(1, 100000)},
        },
        'jrc_nrt':  {
            'run': {'type': bool},
        },
        'global_products': {
            'run': {'type': bool},
            'gfc': {'type': bool},
            'tmf': {'type': bool},
            'tmf_years': {'type': bool},
            'esa_lc20': {'type': bool},
            'copernicus_lc': {'type': bool},
            'esri_lc': {'type': bool},
            'lang_tree_height': {'type': bool},
            'potapov_tree_height': {'type': bool},
            'elevation': {'type': bool},
            'dynamic_world_tree_prob': {'type': bool},
            'dynamic_world_class_mode': {'type': bool},
        },
        'py_workers': {'type': int, 'choices': range(1, 10000)},
        'ee_workers': {'type': int, 'choices': range(1, 10000)},
        'outdir': {'type': str}
    },
    "subsampling_params": {
        "th": {
            "percentile": {'type': int, 'choices': range(1, 99)},
            "tree_cover": {'type': int, 'choices': range(0, 100)},
            "tree_height": {'type': int, 'choices': range(0, 100)},
            "max_points":  {'type': (int, bool), 'choices': range(0, 9999999999999)},
            "random_state": {'type': int, 'choices': range(0, 9999999999999)},
        },
        "outdir": {'type': str}
    }
}
