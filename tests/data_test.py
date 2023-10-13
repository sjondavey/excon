
import pandas as pd


def test_data():
    excon_definitions_and_embeddings_file = "./inputs/definitions_with_embeddings.parquet"
    df_definitions = pd.read_parquet(excon_definitions_and_embeddings_file, engine='pyarrow')
    assert len(df_definitions) == 55

    excon_definitions_and_embeddings_file = "./inputs/definitions_insurance_with_embeddings.parquet"
    df_definitions_insurance = pd.read_parquet(excon_definitions_and_embeddings_file, engine='pyarrow')
    assert len(df_definitions_insurance) == 21

    excon_definitions_and_embeddings_file = "./inputs/definitions_securities_with_embeddings.parquet"
    df_definitions_securities = pd.read_parquet(excon_definitions_and_embeddings_file, engine='pyarrow')
    assert len(df_definitions_securities) == 14
    
    # Load the section headings. For larger files, parquet is better (compressed more and loads faster)
    excon_section_headings_and_embeddings = "./inputs/section_headings_with_embeddings.parquet"
    df_sections = pd.read_parquet(excon_section_headings_and_embeddings, engine='pyarrow')
    assert len(df_sections) == 376

    section_summary_with_embeddings = "./inputs/summary_with_embedding.parquet"
    df_summary = pd.read_parquet(section_summary_with_embeddings, engine='pyarrow')
    assert len(df_summary) == 519

    section_questions_with_embeddings = "./inputs/questions_with_embedding.parquet"
    df_summary_questions = pd.read_parquet(section_questions_with_embeddings, engine='pyarrow')
    assert len(df_summary_questions) == 717
