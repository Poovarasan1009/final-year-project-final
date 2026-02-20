import pandas as pd
from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class BatchEvaluator:
    def __init__(self):
        self.evaluator = AdvancedAnswerEvaluator()
    
    def run(self, dataset_path="Real_Dataset/sample_dataset.csv"):
        """Run batch evaluation on dataset"""
        print(f"📊 Running batch evaluation on {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        results = []
        
        # Evaluate each row
        for idx, row in df.iterrows():
            print(f"  Evaluating answer {idx+1}/{len(df)}...")
            
            result = self.evaluator.evaluate(
                row['question'],
                row['ideal_answer'],
                row['student_answer']
            )
            
            results.append({
                'question': row['question'],
                'student_answer': row['student_answer'][:100] + '...',
                'ai_score': result['final_score'],
                'human_score': row.get('human_score', 'N/A'),
                'difference': abs(result['final_score'] - row.get('human_score', 0)) if 'human_score' in row else 'N/A',
                'feedback': result['feedback']
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        stats = self.calculate_statistics(results_df)
        
        # Generate visualizations
        self.generate_visualizations(results_df)
        
        return {
            'total': len(results_df),
            'average_score': results_df['ai_score'].mean(),
            'statistics': stats,
            'details': results
        }
    
    def calculate_statistics(self, df):
        """Calculate evaluation statistics"""
        stats = {
            'total_answers': len(df),
            'average_score': round(df['ai_score'].mean(), 2),
            'score_distribution': {
                'excellent': len(df[df['ai_score'] >= 80]),
                'good': len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)]),
                'average': len(df[(df['ai_score'] >= 40) & (df['ai_score'] < 60)]),
                'poor': len(df[df['ai_score'] < 40])
            }
        }
        
        # If human scores available, calculate correlation
        if 'human_score' in df.columns and 'difference' in df.columns:
            human_scores = df['human_score'].dropna()
            if len(human_scores) > 0:
                correlation = df[['ai_score', 'human_score']].corr().iloc[0, 1]
                stats['human_ai_correlation'] = round(correlation, 3)
                stats['average_difference'] = round(df['difference'].mean(), 2)
        
        return stats
    
    def generate_visualizations(self, df):
        """Generate evaluation visualizations"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Score distribution
        plt.subplot(1, 3, 1)
        plt.hist(df['ai_score'], bins=10, color='skyblue', edgecolor='black')
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Score comparison (if human scores available)
        if 'human_score' in df.columns:
            plt.subplot(1, 3, 2)
            plt.scatter(df['human_score'], df['ai_score'], alpha=0.6)
            plt.plot([0, 100], [0, 100], 'r--', alpha=0.5)
            plt.title('AI vs Human Scoring')
            plt.xlabel('Human Score')
            plt.ylabel('AI Score')
            plt.grid(True, alpha=0.3)
        
        # Plot 3: Score categories
        plt.subplot(1, 3, 3)
        categories = ['Excellent', 'Good', 'Average', 'Poor']
        counts = [
            len(df[df['ai_score'] >= 80]),
            len(df[(df['ai_score'] >= 60) & (df['ai_score'] < 80)]),
            len(df[(df['ai_score'] >= 40) & (df['ai_score'] < 60)]),
            len(df[df['ai_score'] < 40])
        ]
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        plt.bar(categories, counts, color=colors)
        plt.title('Score Categories')
        plt.xlabel('Category')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('Results/batch_evaluation_results.png', dpi=150)
        plt.close()
        
        print("✓ Visualizations saved to Results/batch_evaluation_results.png")