"""
Mass Claims Analyzer for Sablemoore Analytics
Specialized for high-volume standardized claims (PCP, PPI, etc.)

UK Mass Claims Markets:
- PCP (Personal Contract Purchase) mis-selling: £40B+ exposure
- PPI (Payment Protection Insurance): £50B+ paid out
- Diesel emissions claims: £5B+ potential
- Holiday sickness claims: £1B+ market
- Data breach claims (GDPR): Growing market
- Package holiday claims: £500M+ annual
- Flight delay compensation: £300M+ annual

Key Differences from Traditional Litigation:
- Volume: 1,000s-100,000s of claims vs individual cases
- Standardization: Template-based vs bespoke
- Success rates: 60-95% vs 50-70%
- Settlement patterns: Predictable vs negotiated
- Time to resolution: 3-12 months vs 12-36 months
- Claim values: £500-£10,000 vs £10K-£10M+
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False


class MassClaimsAnalyzer:
    """
    Specialized analyzer for mass claims campaigns

    Use Cases:
    1. PCP mis-selling claims (car finance)
    2. Package holiday claims
    3. Flight delay compensation
    4. Data breach claims
    5. Diesel emissions claims
    """

    def __init__(self):
        self.claim_type_models = {}
        self.label_encoders = {}

        # UK mass claims statistics (2020-2025)
        self.mass_claims_stats = {
            'PCP_Car_Finance': {
                'avg_success_rate': 0.78,
                'avg_settlement': 2850,
                'median_duration_days': 180,
                'volume': 'Very High (100,000s)',
                'defender_types': ['Banks', 'Car Dealers', 'Finance Companies'],
                'key_factors': ['Undisclosed commission', 'Unfair relationship', 'Poor explanation'],
                'regulatory_wind': 'Strong (FCA investigation)',
                'market_size_gbp': 40_000_000_000
            },
            'PPI_Mis_Selling': {
                'avg_success_rate': 0.85,
                'avg_settlement': 1750,
                'median_duration_days': 120,
                'volume': 'Declining (deadline passed 2019)',
                'defender_types': ['Banks', 'Credit Card Companies'],
                'key_factors': ['Never told about PPI', 'Self-employed rejection', 'Pre-existing conditions'],
                'regulatory_wind': 'Complete (£50B paid out)',
                'market_size_gbp': 50_000_000_000
            },
            'Diesel_Emissions': {
                'avg_success_rate': 0.65,
                'avg_settlement': 3200,
                'median_duration_days': 240,
                'volume': 'High (100,000s)',
                'defender_types': ['VW Group', 'Mercedes', 'BMW', 'Vauxhall'],
                'key_factors': ['Defeat device', 'Emission test cheating', 'Reduced value'],
                'regulatory_wind': 'Medium (Class actions ongoing)',
                'market_size_gbp': 5_000_000_000
            },
            'Package_Holiday': {
                'avg_success_rate': 0.72,
                'avg_settlement': 1850,
                'median_duration_days': 90,
                'volume': 'Medium (10,000s)',
                'defender_types': ['Tour Operators', 'Hotels', 'Airlines'],
                'key_factors': ['Illness', 'False description', 'Cancellation'],
                'regulatory_wind': 'Neutral (Package Travel Regulations)',
                'market_size_gbp': 500_000_000
            },
            'Flight_Delay': {
                'avg_success_rate': 0.88,
                'avg_settlement': 420,
                'median_duration_days': 60,
                'volume': 'Very High (100,000s)',
                'defender_types': ['Airlines'],
                'key_factors': ['3+ hour delay', 'Not extraordinary circumstances', 'EU261 applies'],
                'regulatory_wind': 'Strong (EU Regulation 261/2004)',
                'market_size_gbp': 300_000_000
            },
            'Holiday_Sickness': {
                'avg_success_rate': 0.45,
                'avg_settlement': 2500,
                'median_duration_days': 150,
                'volume': 'Declining (fraud crackdown)',
                'defender_types': ['Tour Operators', 'Hotels'],
                'key_factors': ['Genuine illness', 'Medical evidence', 'No fraud indicators'],
                'regulatory_wind': 'Negative (Courts cracking down)',
                'market_size_gbp': 200_000_000
            },
            'Data_Breach_GDPR': {
                'avg_success_rate': 0.55,
                'avg_settlement': 850,
                'median_duration_days': 180,
                'volume': 'Growing (10,000s)',
                'defender_types': ['Tech Companies', 'Retailers', 'Public Bodies'],
                'key_factors': ['Data breach confirmed', 'Personal data involved', 'Distress caused'],
                'regulatory_wind': 'Strong (GDPR enforcement)',
                'market_size_gbp': 1_000_000_000
            }
        }

    def generate_mass_claims_dataset(self, claim_type='PCP_Car_Finance', num_claims=10000):
        """
        Generate realistic mass claims dataset for training

        Args:
            claim_type: Type of mass claim
            num_claims: Number of claims to generate
        """
        print(f"\n🔬 Generating {num_claims:,} {claim_type} claims...")

        stats = self.mass_claims_stats.get(claim_type, self.mass_claims_stats['PCP_Car_Finance'])

        data = []

        for i in range(num_claims):
            # Claim characteristics
            claim_date = datetime.now() - timedelta(days=np.random.randint(0, 730))  # Last 2 years

            # Claimant characteristics
            claimant_type = np.random.choice(['Individual', 'Small Business'], p=[0.95, 0.05])

            # Legal representation
            has_lawyer = np.random.choice([True, False], p=[0.85, 0.15])  # 85% use CMCs or lawyers

            # Evidence quality
            evidence_quality = np.random.choice(['Strong', 'Medium', 'Weak'], p=[0.30, 0.50, 0.20])

            # Claim specifics
            if claim_type == 'PCP_Car_Finance':
                claim_amount = np.random.lognormal(7.8, 0.4)  # £1,500-£5,000
                claim_amount = np.clip(claim_amount, 500, 15000)

                finance_company = np.random.choice(['Barclays', 'Santander', 'HSBC', 'BlackHorse', 'BMW Finance', 'VW Finance'])
                commission_rate = np.random.uniform(0.15, 0.35)  # 15-35% undisclosed commission

                # Success factors
                success_rate = stats['avg_success_rate']

                if commission_rate > 0.28:  # Very high commission
                    success_rate *= 1.15
                if evidence_quality == 'Strong':
                    success_rate *= 1.10
                elif evidence_quality == 'Weak':
                    success_rate *= 0.75
                if has_lawyer:
                    success_rate *= 1.08

            elif claim_type == 'Flight_Delay':
                # EU261 compensation tiers
                distance = np.random.choice(['<1500km', '1500-3500km', '>3500km'], p=[0.50, 0.35, 0.15])
                if distance == '<1500km':
                    claim_amount = 250
                elif distance == '1500-3500km':
                    claim_amount = 400
                else:
                    claim_amount = 600

                delay_hours = np.random.uniform(3, 12)
                extraordinary = np.random.choice([True, False], p=[0.12, 0.88])  # 12% extraordinary circumstances

                success_rate = stats['avg_success_rate']

                if extraordinary:
                    success_rate *= 0.05  # Almost no chance if extraordinary circumstances
                if delay_hours < 3.5:
                    success_rate *= 0.30  # Borderline 3 hours
                if evidence_quality == 'Strong':
                    success_rate *= 1.05

            elif claim_type == 'Diesel_Emissions':
                claim_amount = np.random.lognormal(8.0, 0.3)
                claim_amount = np.clip(claim_amount, 1000, 8000)

                manufacturer = np.random.choice(['VW', 'Audi', 'Skoda', 'SEAT', 'Mercedes', 'BMW'], p=[0.35, 0.20, 0.15, 0.10, 0.12, 0.08])
                vehicle_age = np.random.randint(3, 12)  # 3-12 years old

                success_rate = stats['avg_success_rate']

                if manufacturer in ['VW', 'Audi']:  # Proven cheating
                    success_rate *= 1.20
                if vehicle_age > 8:
                    success_rate *= 0.90  # Harder to prove loss
                if evidence_quality == 'Strong':
                    success_rate *= 1.15

            else:
                # Generic mass claim
                claim_amount = np.random.lognormal(7.5, 0.5)
                claim_amount = np.clip(claim_amount, 500, 10000)
                success_rate = stats['avg_success_rate']

            # Add realistic noise
            success_rate += np.random.normal(0, 0.05)
            success_rate = np.clip(success_rate, 0.05, 0.98)

            # Determine outcome
            outcome = 1 if np.random.random() < success_rate else 0

            # Settlement amount (if successful)
            if outcome == 1:
                settlement = claim_amount * np.random.uniform(0.70, 1.10)  # 70-110% of claim
            else:
                settlement = 0

            # Duration
            duration_days = int(stats['median_duration_days'] * np.random.uniform(0.6, 1.8))

            data.append({
                'claim_type': claim_type,
                'claim_date': claim_date.strftime('%Y-%m-%d'),
                'claimant_type': claimant_type,
                'has_legal_rep': has_lawyer,
                'evidence_quality': evidence_quality,
                'claim_amount': claim_amount,
                'success_rate': success_rate,
                'outcome': outcome,
                'settlement_amount': settlement,
                'duration_days': duration_days,
                'year': claim_date.year
            })

        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df):,} claims")
        print(f"   Success rate: {df['outcome'].mean()*100:.1f}%")
        print(f"   Avg settlement: £{df[df['outcome']==1]['settlement_amount'].mean():.0f}")

        return df

    def train_mass_claims_model(self, claim_types=['PCP_Car_Finance', 'Flight_Delay', 'Diesel_Emissions']):
        """
        Train specialized models for each mass claim type
        """
        print("\n" + "="*70)
        print("TRAINING MASS CLAIMS MODELS")
        print("="*70)

        all_data = []

        for claim_type in claim_types:
            print(f"\n📊 Processing {claim_type}...")
            df = self.generate_mass_claims_dataset(claim_type, num_claims=20000)
            all_data.append(df)

        # Combine all claim types
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Total dataset: {len(combined_df):,} claims across {len(claim_types)} types")

        # Encode categorical features
        categorical_cols = ['claim_type', 'claimant_type', 'evidence_quality']

        for col in categorical_cols:
            le = LabelEncoder()
            combined_df[f'{col}_enc'] = le.fit_transform(combined_df[col])
            self.label_encoders[col] = le

        # Features
        combined_df['claim_amount_log'] = np.log1p(combined_df['claim_amount'])
        combined_df['has_legal_rep_int'] = combined_df['has_legal_rep'].astype(int)
        combined_df['high_value'] = (combined_df['claim_amount'] > 2000).astype(int)

        feature_cols = ['claim_type_enc', 'claimant_type_enc', 'evidence_quality_enc',
                       'has_legal_rep_int', 'claim_amount_log', 'high_value']

        X = combined_df[feature_cols].values
        y = combined_df['outcome'].values

        # Train model
        print(f"\n🤖 Training mass claims classifier...")

        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
            print("   Using: XGBoost")
        else:
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                random_state=42
            )
            print("   Using: Gradient Boosting")

        model.fit(X, y)

        # Evaluate
        predictions = model.predict(X)
        accuracy = (predictions == y).mean()

        print(f"✓ Model trained")
        print(f"   Accuracy: {accuracy*100:.2f}%")

        self.mass_claims_model = model
        self.feature_cols = feature_cols

        # Save model
        model_data = {
            'model': model,
            'label_encoders': self.label_encoders,
            'feature_cols': feature_cols,
            'claim_stats': self.mass_claims_stats,
            'training_date': datetime.now().strftime('%Y-%m-%d')
        }

        with open('mass_claims_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Saved: mass_claims_model.pkl")

        return accuracy

    def predict_mass_claim(self, claim_data):
        """
        Predict success and settlement for a mass claim

        Args:
            claim_data: dict with keys:
                - claim_type: 'PCP_Car_Finance', 'Flight_Delay', etc.
                - claimant_type: 'Individual' or 'Small Business'
                - evidence_quality: 'Strong', 'Medium', 'Weak'
                - has_legal_rep: True/False
                - claim_amount: float
        """
        # Encode features
        features = {}

        for col in ['claim_type', 'claimant_type', 'evidence_quality']:
            value = claim_data.get(col)
            if value in self.label_encoders[col].classes_:
                features[f'{col}_enc'] = self.label_encoders[col].transform([value])[0]
            else:
                features[f'{col}_enc'] = 0

        features['has_legal_rep_int'] = int(claim_data.get('has_legal_rep', True))
        features['claim_amount_log'] = np.log1p(claim_data.get('claim_amount', 2000))
        features['high_value'] = int(claim_data.get('claim_amount', 2000) > 2000)

        X = np.array([[features[col] for col in self.feature_cols]])

        # Predict
        success_prob = self.mass_claims_model.predict_proba(X)[0][1]

        # Get stats for claim type
        stats = self.mass_claims_stats.get(claim_data['claim_type'], {})

        # Estimate settlement
        if success_prob > 0.5:
            settlement = claim_data.get('claim_amount', 2000) * np.random.uniform(0.75, 1.05)
        else:
            settlement = 0

        # Duration estimate
        median_duration = stats.get('median_duration_days', 180)

        return {
            'success_probability': success_prob,
            'predicted_outcome': 'Success' if success_prob > 0.5 else 'Rejection',
            'confidence': 'High' if abs(success_prob - 0.5) > 0.3 else 'Medium',
            'estimated_settlement': settlement,
            'estimated_duration_days': median_duration,
            'estimated_duration_months': median_duration / 30,
            'market_stats': stats,
            'recommendation': self._generate_mass_claim_recommendation(success_prob, claim_data)
        }

    def _generate_mass_claim_recommendation(self, success_prob, claim_data):
        """Generate investment recommendation for mass claims"""

        claim_amount = claim_data.get('claim_amount', 2000)

        if success_prob > 0.80:
            roi = claim_amount * 0.25  # Typical 25% of settlement
            return f"STRONG PURSUE - {success_prob*100:.0f}% success. Est. ROI: £{roi:.0f} per claim"
        elif success_prob > 0.65:
            roi = claim_amount * 0.20
            return f"PURSUE - Good odds. Est. ROI: £{roi:.0f} per claim"
        elif success_prob > 0.45:
            return f"MARGINAL - {success_prob*100:.0f}% success. Consider portfolio approach"
        else:
            return f"AVOID - Low success probability ({success_prob*100:.0f}%)"

    def analyze_portfolio(self, claims_df):
        """
        Analyze a portfolio of mass claims

        Returns portfolio-level metrics for investment decision
        """
        print("\n" + "="*70)
        print("MASS CLAIMS PORTFOLIO ANALYSIS")
        print("="*70)

        # Predict each claim
        predictions = []

        for idx, claim in claims_df.iterrows():
            pred = self.predict_mass_claim(claim.to_dict())
            predictions.append(pred)

        pred_df = pd.DataFrame(predictions)

        # Portfolio metrics
        total_claims = len(claims_df)
        expected_successes = pred_df['success_probability'].sum()
        total_claimed = claims_df['claim_amount'].sum()
        expected_settlements = pred_df['estimated_settlement'].sum()

        # ROI calculation (assuming 25% fee)
        gross_revenue = expected_settlements * 0.25

        # Costs (estimate £150 per claim for CMC)
        total_costs = total_claims * 150

        net_profit = gross_revenue - total_costs
        roi = (net_profit / total_costs) * 100 if total_costs > 0 else 0

        print(f"\n📊 Portfolio Summary:")
        print(f"   Total claims: {total_claims:,}")
        print(f"   Expected successes: {expected_successes:.0f} ({expected_successes/total_claims*100:.1f}%)")
        print(f"   Total claimed: £{total_claimed:,.0f}")
        print(f"   Expected settlements: £{expected_settlements:,.0f}")
        print(f"\n💰 Financial Projections:")
        print(f"   Gross revenue (25% fee): £{gross_revenue:,.0f}")
        print(f"   Total costs (£150/claim): £{total_costs:,.0f}")
        print(f"   Net profit: £{net_profit:,.0f}")
        print(f"   ROI: {roi:.1f}%")

        return {
            'total_claims': total_claims,
            'expected_successes': expected_successes,
            'success_rate': expected_successes / total_claims,
            'total_claimed': total_claimed,
            'expected_settlements': expected_settlements,
            'gross_revenue': gross_revenue,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'roi_percent': roi
        }


def demo_mass_claims():
    """
    Demonstration of mass claims analysis
    """
    print("\n" + "="*70)
    print("SABLEMOORE MASS CLAIMS ANALYZER - DEMO")
    print("UK Mass Claims Intelligence System")
    print("="*70)

    analyzer = MassClaimsAnalyzer()

    # Train models
    print("\n[1/3] TRAINING MODELS")
    accuracy = analyzer.train_mass_claims_model([
        'PCP_Car_Finance',
        'Flight_Delay',
        'Diesel_Emissions',
        'Package_Holiday'
    ])

    # Example predictions
    print("\n[2/3] EXAMPLE PREDICTIONS")

    # PCP claim
    print("\n📋 Example 1: PCP Car Finance Claim")
    pcp_claim = {
        'claim_type': 'PCP_Car_Finance',
        'claimant_type': 'Individual',
        'evidence_quality': 'Strong',
        'has_legal_rep': True,
        'claim_amount': 3500
    }

    result = analyzer.predict_mass_claim(pcp_claim)
    print(f"   Success probability: {result['success_probability']*100:.1f}%")
    print(f"   Est. settlement: £{result['estimated_settlement']:.0f}")
    print(f"   Duration: {result['estimated_duration_months']:.1f} months")
    print(f"   Recommendation: {result['recommendation']}")

    # Flight delay
    print("\n📋 Example 2: Flight Delay Claim")
    flight_claim = {
        'claim_type': 'Flight_Delay',
        'claimant_type': 'Individual',
        'evidence_quality': 'Strong',
        'has_legal_rep': False,
        'claim_amount': 400
    }

    result = analyzer.predict_mass_claim(flight_claim)
    print(f"   Success probability: {result['success_probability']*100:.1f}%")
    print(f"   Est. settlement: £{result['estimated_settlement']:.0f}")
    print(f"   Duration: {result['estimated_duration_months']:.1f} months")
    print(f"   Recommendation: {result['recommendation']}")

    # Portfolio analysis
    print("\n[3/3] PORTFOLIO ANALYSIS")
    print("\n📦 Sample Portfolio: 1,000 PCP Claims")

    # Generate sample portfolio
    portfolio_df = analyzer.generate_mass_claims_dataset('PCP_Car_Finance', 1000)
    portfolio_metrics = analyzer.analyze_portfolio(portfolio_df)

    print(f"\n✅ DEMO COMPLETE")
    print(f"\n💡 Key Insight:")
    print(f"   A portfolio of 1,000 PCP claims with {portfolio_metrics['success_rate']*100:.0f}% success rate")
    print(f"   generates £{portfolio_metrics['net_profit']:,.0f} net profit")
    print(f"   ({portfolio_metrics['roi_percent']:.0f}% ROI)")


if __name__ == "__main__":
    demo_mass_claims()
