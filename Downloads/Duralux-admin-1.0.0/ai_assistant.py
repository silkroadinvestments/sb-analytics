"""
Sablemoore Analytics - AI Assistant Module
Provides intelligent analysis, explanations, and verification for litigation cases.
Supports multiple LLM backends: OpenAI, Anthropic, or local rule-based fallback.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# Try to import LLM libraries
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass


class AIAssistant:
    """
    AI-powered assistant for litigation finance analysis.
    Provides natural language explanations, verification, and insights.
    """

    # UK Legal knowledge base for rule-based fallback
    UK_LEGAL_KNOWLEDGE = {
        'case_types': {
            'Contract Dispute': {
                'description': 'Disputes arising from breach of contract, non-performance, or contractual interpretation',
                'typical_success_factors': ['Clear written contract', 'Documented breach', 'Quantifiable damages'],
                'common_risks': ['Ambiguous contract terms', 'Limitation periods', 'Mitigation requirements'],
                'avg_duration': '12-18 months',
                'key_legislation': ['Contract Act 1999', 'Sale of Goods Act 1979']
            },
            'Personal Injury': {
                'description': 'Claims for compensation due to physical or psychological injury caused by negligence',
                'typical_success_factors': ['Clear liability', 'Medical evidence', 'Causation established'],
                'common_risks': ['Contributory negligence', 'Pre-existing conditions', 'Limitation periods'],
                'avg_duration': '18-36 months',
                'key_legislation': ['Limitation Act 1980', 'Civil Liability (Contribution) Act 1978']
            },
            'Employment': {
                'description': 'Disputes between employers and employees including unfair dismissal and discrimination',
                'typical_success_factors': ['Documented misconduct by employer', 'Following grievance procedures', 'Witness evidence'],
                'common_risks': ['Time limits (3 months)', 'Caps on compensation', 'Employer documentation'],
                'avg_duration': '6-12 months',
                'key_legislation': ['Employment Rights Act 1996', 'Equality Act 2010']
            },
            'Commercial Dispute': {
                'description': 'Business-to-business disputes involving partnerships, shareholders, or commercial agreements',
                'typical_success_factors': ['Clear documentation', 'Expert accounting evidence', 'Defined losses'],
                'common_risks': ['Complex multi-party issues', 'Costs exposure', 'Business relationship damage'],
                'avg_duration': '18-30 months',
                'key_legislation': ['Companies Act 2006', 'Partnership Act 1890']
            },
            'Professional Negligence': {
                'description': 'Claims against professionals (solicitors, accountants, surveyors) for failing to meet standards',
                'typical_success_factors': ['Expert evidence of breach', 'Clear causation', 'Quantifiable loss'],
                'common_risks': ['High evidential burden', 'Complex causation issues', 'Limitation periods'],
                'avg_duration': '24-36 months',
                'key_legislation': ['Supply of Services Act 1982', 'Bolam test principles']
            },
            'Fraud': {
                'description': 'Civil claims arising from fraudulent misrepresentation or deceit',
                'typical_success_factors': ['Clear evidence of dishonesty', 'Reliance on misrepresentation', 'Quantifiable loss'],
                'common_risks': ['High burden of proof', 'Asset tracing difficulties', 'Defendant insolvency'],
                'avg_duration': '24-48 months',
                'key_legislation': ['Fraud Act 2006', 'Limitation Act 1980 s.32']
            },
            'Debt Recovery': {
                'description': 'Claims to recover monies owed under contracts or agreements',
                'typical_success_factors': ['Clear debt documentation', 'No valid defence', 'Solvent defendant'],
                'common_risks': ['Defendant insolvency', 'Set-off claims', 'Enforcement challenges'],
                'avg_duration': '3-9 months',
                'key_legislation': ['Late Payment of Commercial Debts Act 1998']
            }
        },
        'jurisdictions': {
            'County Court': 'Handles claims up to £100,000 (or £10,000 small claims). More cost-effective, faster resolution.',
            'High Court': 'Complex cases over £100,000. More formal procedures, higher costs, specialist judges.',
            'Employment Tribunal': 'Employment disputes only. No costs shifting usually. Strict time limits.',
            'Court of Appeal': 'Appeals on points of law. Permission required. Lower success rates.',
            'Supreme Court': 'Final appellate court. Only cases of general public importance.'
        },
        'risk_factors': {
            'high_value': 'High value claims (>£1M) typically face more vigorous defence and longer litigation',
            'multiple_parties': 'Multiple defendants can complicate proceedings and increase costs',
            'expert_evidence': 'Cases requiring expert evidence are more expensive but experts strengthen cases',
            'appellate': 'Appellate proceedings have lower success rates as courts defer to first instance findings',
            'limitation': 'Cases close to limitation periods require urgent action'
        }
    }

    def __init__(self, api_key: Optional[str] = None, provider: str = 'auto'):
        """
        Initialize AI Assistant.

        Args:
            api_key: API key for LLM provider (or set via environment variable)
            provider: 'openai', 'anthropic', or 'auto' (tries both, falls back to rules)
        """
        self.provider = provider
        self.api_key = api_key
        self.client = None
        self.model = None

        # Try to initialize LLM client
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize the LLM client based on available libraries and API keys"""
        if self.provider == 'auto':
            # Try Anthropic first (Claude), then OpenAI
            if ANTHROPIC_AVAILABLE:
                key = self.api_key or os.environ.get('ANTHROPIC_API_KEY')
                if key:
                    try:
                        self.client = anthropic.Anthropic(api_key=key)
                        self.provider = 'anthropic'
                        self.model = 'claude-3-haiku-20240307'
                        print("[AI Assistant] Initialized with Anthropic Claude")
                        return
                    except Exception as e:
                        print(f"[AI Assistant] Anthropic init failed: {e}")

            if OPENAI_AVAILABLE:
                key = self.api_key or os.environ.get('OPENAI_API_KEY')
                if key:
                    try:
                        self.client = openai.OpenAI(api_key=key)
                        self.provider = 'openai'
                        self.model = 'gpt-3.5-turbo'
                        print("[AI Assistant] Initialized with OpenAI GPT")
                        return
                    except Exception as e:
                        print(f"[AI Assistant] OpenAI init failed: {e}")

            # Fall back to rule-based
            self.provider = 'rules'
            print("[AI Assistant] Using rule-based analysis (no LLM API available)")

        elif self.provider == 'anthropic' and ANTHROPIC_AVAILABLE:
            key = self.api_key or os.environ.get('ANTHROPIC_API_KEY')
            if key:
                self.client = anthropic.Anthropic(api_key=key)
                self.model = 'claude-3-haiku-20240307'

        elif self.provider == 'openai' and OPENAI_AVAILABLE:
            key = self.api_key or os.environ.get('OPENAI_API_KEY')
            if key:
                self.client = openai.OpenAI(api_key=key)
                self.model = 'gpt-3.5-turbo'

    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call the LLM API"""
        if not self.client:
            return None

        try:
            if self.provider == 'anthropic':
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_prompt or "You are a UK litigation finance expert assistant.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text

            elif self.provider == 'openai':
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1024
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"[AI Assistant] LLM call failed: {e}")
            return None

    def analyze_case(self, case_data: Dict, prediction: Dict) -> Dict:
        """
        Provide AI-powered analysis and explanation of a case.

        Returns detailed explanation of the case characteristics,
        success factors, risks, and recommendations.
        """
        case_type = case_data.get('case_type', 'General Civil Litigation')
        jurisdiction = case_data.get('jurisdiction', 'County Court')
        claim_amount = case_data.get('claim_amount', 0)
        complexity = case_data.get('complexity', 'Medium')
        defendant_type = case_data.get('defendant_type', 'Individual')
        success_rate = prediction.get('success_rate', 50)
        risk_level = prediction.get('risk_level', 'Medium')

        # Try LLM first
        if self.client:
            prompt = f"""Analyze this UK litigation case for funding potential:

Case Type: {case_type}
Jurisdiction: {jurisdiction}
Claim Amount: £{claim_amount:,.0f}
Complexity: {complexity}
Defendant Type: {defendant_type}
ML Predicted Success Rate: {success_rate}%
Risk Level: {risk_level}

Case Summary: {case_data.get('raw_text', 'No summary available')[:500]}

Provide a concise analysis (3-4 paragraphs) covering:
1. Key strengths and weaknesses of this case
2. Specific risks to consider for litigation funding
3. Recommendation on funding viability
4. Any red flags or concerns

Focus on practical, actionable insights for litigation funders."""

            system = """You are an expert UK litigation finance analyst.
Provide clear, professional analysis focusing on funding viability.
Use UK legal terminology and reference relevant legislation where appropriate.
Be direct and practical - funders need actionable insights."""

            llm_response = self._call_llm(prompt, system)
            if llm_response:
                return {
                    'analysis': llm_response,
                    'source': 'ai',
                    'model': self.model,
                    'timestamp': datetime.now().isoformat()
                }

        # Fall back to rule-based analysis
        return self._rule_based_case_analysis(case_data, prediction)

    def _rule_based_case_analysis(self, case_data: Dict, prediction: Dict) -> Dict:
        """Generate rule-based case analysis when LLM is not available"""
        case_type = case_data.get('case_type', 'General Civil Litigation')
        jurisdiction = case_data.get('jurisdiction', 'County Court')
        claim_amount = case_data.get('claim_amount', 0)
        complexity = case_data.get('complexity', 'Medium')
        defendant_type = case_data.get('defendant_type', 'Individual')
        success_rate = prediction.get('success_rate', 50)
        risk_level = prediction.get('risk_level', 'Medium')

        # Get knowledge base info
        type_info = self.UK_LEGAL_KNOWLEDGE['case_types'].get(case_type, {})
        jurisdiction_info = self.UK_LEGAL_KNOWLEDGE['jurisdictions'].get(jurisdiction, '')

        # Build analysis
        analysis_parts = []

        # Case overview
        overview = f"**Case Overview:** This is a {case_type} case in the {jurisdiction}"
        if type_info.get('description'):
            overview += f". {type_info['description']}"
        overview += f" The claim value of £{claim_amount:,.0f} "
        if claim_amount > 1000000:
            overview += "places this in the high-value category, which typically attracts vigorous defence."
        elif claim_amount > 100000:
            overview += "is substantial and warrants careful due diligence."
        else:
            overview += "is moderate, which may support cost-effective resolution."
        analysis_parts.append(overview)

        # Success factors
        if type_info.get('typical_success_factors'):
            factors = type_info['typical_success_factors']
            analysis_parts.append(f"**Key Success Factors:** For {case_type} cases, success typically depends on: {', '.join(factors)}. Review the case documents to confirm these elements are present.")

        # Risks
        risks = []
        if complexity == 'High':
            risks.append("high complexity increases uncertainty and costs")
        if defendant_type == 'Corporate':
            risks.append("corporate defendants often have substantial resources for defence")
        if defendant_type == 'Public Body':
            risks.append("public body defendants may have policy constraints affecting settlement")
        if type_info.get('common_risks'):
            risks.extend(type_info['common_risks'][:2])

        if risks:
            analysis_parts.append(f"**Risk Factors:** Key risks include: {'; '.join(risks)}.")

        # Recommendation
        if success_rate >= 70:
            rec = f"**Recommendation:** With a predicted success rate of {success_rate}%, this case shows strong funding potential. "
            rec += "Recommend proceeding with detailed due diligence."
        elif success_rate >= 55:
            rec = f"**Recommendation:** The predicted success rate of {success_rate}% indicates moderate potential. "
            rec += "Consider requesting additional documentation to verify key assumptions."
        else:
            rec = f"**Recommendation:** The predicted success rate of {success_rate}% suggests elevated risk. "
            rec += "Recommend careful scrutiny of the weaknesses identified before proceeding."

        if risk_level == 'High':
            rec += " Note: Risk level is HIGH - ensure appropriate risk premium in funding terms."

        analysis_parts.append(rec)

        # Jurisdiction note
        if jurisdiction_info:
            analysis_parts.append(f"**Jurisdiction Note:** {jurisdiction_info}")

        return {
            'analysis': '\n\n'.join(analysis_parts),
            'source': 'rules',
            'model': 'rule-based',
            'timestamp': datetime.now().isoformat()
        }

    def verify_duplicate(self, case1: Dict, case2: Dict, similarity_score: float,
                        features: Dict) -> Dict:
        """
        AI verification of potential duplicate cases.
        Explains why cases were flagged and confidence in the match.
        """
        # Extract key info
        c1_type = case1.get('case_type', 'Unknown')
        c2_type = case2.get('case_type', 'Unknown')
        c1_amount = case1.get('claim_amount', 0)
        c2_amount = case2.get('claim_amount', 0)

        text_sim = features.get('text_similarity', 0)
        name_match = features.get('defendant_name_match', 0)
        claimant_match = features.get('claimant_name_match', 0)

        # Try LLM
        if self.client:
            prompt = f"""Analyze these two potentially duplicate litigation cases:

CASE 1:
- Type: {c1_type}
- Claim: £{c1_amount:,.0f}
- Summary: {case1.get('raw_text', '')[:300]}

CASE 2:
- Type: {c2_type}
- Claim: £{c2_amount:,.0f}
- Summary: {case2.get('raw_text', '')[:300]}

SIMILARITY METRICS:
- Overall similarity: {similarity_score}%
- Text similarity: {text_sim}%
- Defendant name match: {name_match}%
- Claimant name match: {claimant_match}%

Provide:
1. Your assessment: Are these likely duplicates, related cases, or false positive?
2. Key evidence supporting your conclusion
3. Confidence level (High/Medium/Low)
4. Recommended action

Be concise but thorough."""

            system = "You are an expert at identifying duplicate litigation cases. Be analytical and precise."

            llm_response = self._call_llm(prompt, system)
            if llm_response:
                return {
                    'verification': llm_response,
                    'source': 'ai',
                    'model': self.model,
                    'timestamp': datetime.now().isoformat()
                }

        # Rule-based verification
        return self._rule_based_duplicate_verification(case1, case2, similarity_score, features)

    def _rule_based_duplicate_verification(self, case1: Dict, case2: Dict,
                                          similarity_score: float, features: Dict) -> Dict:
        """Rule-based duplicate verification"""
        reasons = []
        confidence = 'Low'
        likely_duplicate = False

        text_sim = features.get('text_similarity', 0)
        name_match = features.get('defendant_name_match', 0)
        claimant_match = features.get('claimant_name_match', 0)
        same_type = features.get('same_type', False)
        amount_match = features.get('amount_match', 0)

        # Analyze evidence
        if name_match >= 80 and claimant_match >= 80:
            reasons.append(f"Strong party name match - both defendant ({name_match}%) and claimant ({claimant_match}%) names are highly similar")
            confidence = 'High'
            likely_duplicate = True
        elif name_match >= 70 or claimant_match >= 70:
            reasons.append(f"Significant party name overlap detected (Defendant: {name_match}%, Claimant: {claimant_match}%)")
            confidence = 'Medium'

        if text_sim >= 70:
            reasons.append(f"High text similarity ({text_sim}%) suggests similar or identical case descriptions")
            if confidence != 'High':
                confidence = 'Medium'
            likely_duplicate = True
        elif text_sim >= 50:
            reasons.append(f"Moderate text overlap ({text_sim}%) - cases may be related")

        if same_type and amount_match >= 90:
            reasons.append(f"Same case type with very similar claim amounts ({amount_match}% match)")
            likely_duplicate = True

        if not same_type:
            reasons.append("Different case types - less likely to be true duplicates")
            if confidence == 'High':
                confidence = 'Medium'

        # Build assessment
        if likely_duplicate and confidence == 'High':
            assessment = "**LIKELY DUPLICATE** - Multiple strong indicators suggest these are the same case."
            action = "Recommend merging or flagging for manual review before processing."
        elif likely_duplicate:
            assessment = "**POSSIBLE DUPLICATE** - Some indicators suggest these cases may be related or duplicates."
            action = "Recommend manual review to confirm relationship between cases."
        else:
            assessment = "**LIKELY FALSE POSITIVE** - Limited evidence of true duplication."
            action = "Cases can likely be treated as separate matters, but verify if uncertain."

        verification = f"{assessment}\n\n"
        verification += "**Evidence:**\n" + "\n".join(f"• {r}" for r in reasons)
        verification += f"\n\n**Confidence:** {confidence}\n\n**Recommended Action:** {action}"

        return {
            'verification': verification,
            'likely_duplicate': likely_duplicate,
            'confidence': confidence,
            'source': 'rules',
            'model': 'rule-based',
            'timestamp': datetime.now().isoformat()
        }

    def explain_risk_factors(self, case_data: Dict, prediction: Dict) -> Dict:
        """
        Provide detailed explanation of risk factors and their impact.
        """
        risk_factors = prediction.get('risk_factors', [])
        positive_factors = prediction.get('positive_factors', [])
        success_rate = prediction.get('success_rate', 50)

        if self.client:
            prompt = f"""Explain the risk assessment for this litigation funding opportunity:

Success Rate: {success_rate}%
Risk Level: {prediction.get('risk_level', 'Medium')}

Risk Factors Identified:
{chr(10).join('- ' + r for r in risk_factors) if risk_factors else '- None identified'}

Positive Factors:
{chr(10).join('- ' + p for p in positive_factors) if positive_factors else '- None identified'}

Case Details:
- Type: {case_data.get('case_type', 'Unknown')}
- Claim: £{case_data.get('claim_amount', 0):,.0f}
- Jurisdiction: {case_data.get('jurisdiction', 'Unknown')}
- Complexity: {case_data.get('complexity', 'Unknown')}

Provide:
1. Detailed explanation of each risk factor and its potential impact
2. How the positive factors might mitigate risks
3. Overall risk-adjusted assessment
4. Suggested risk mitigation strategies

Be specific and actionable."""

            system = "You are a UK litigation risk analyst. Provide clear, practical risk analysis for funders."

            llm_response = self._call_llm(prompt, system)
            if llm_response:
                return {
                    'explanation': llm_response,
                    'source': 'ai',
                    'model': self.model,
                    'timestamp': datetime.now().isoformat()
                }

        # Rule-based explanation
        return self._rule_based_risk_explanation(case_data, prediction)

    def _rule_based_risk_explanation(self, case_data: Dict, prediction: Dict) -> Dict:
        """Rule-based risk factor explanation"""
        risk_factors = prediction.get('risk_factors', [])
        positive_factors = prediction.get('positive_factors', [])
        success_rate = prediction.get('success_rate', 50)
        risk_level = prediction.get('risk_level', 'Medium')

        explanation_parts = []

        # Overall assessment
        if risk_level == 'Low':
            explanation_parts.append(f"**Overall Assessment:** This case presents a LOW risk profile with {success_rate}% predicted success. The risk-reward ratio appears favorable for funding.")
        elif risk_level == 'Medium':
            explanation_parts.append(f"**Overall Assessment:** This case has a MEDIUM risk profile with {success_rate}% predicted success. Standard due diligence and pricing should apply.")
        else:
            explanation_parts.append(f"**Overall Assessment:** This case carries HIGH risk with only {success_rate}% predicted success. Enhanced scrutiny and risk premium pricing recommended.")

        # Risk factors detail
        if risk_factors:
            explanation_parts.append("**Risk Factor Analysis:**")
            for factor in risk_factors:
                detail = self._get_risk_detail(factor)
                explanation_parts.append(f"• {factor}: {detail}")
        else:
            explanation_parts.append("**Risk Factors:** No significant risk factors identified in initial analysis.")

        # Positive factors
        if positive_factors:
            explanation_parts.append("\n**Mitigating Factors:**")
            for factor in positive_factors:
                explanation_parts.append(f"• {factor}")

        # Mitigation strategies
        explanation_parts.append("\n**Risk Mitigation Strategies:**")
        if case_data.get('complexity') == 'High':
            explanation_parts.append("• Consider staged funding with milestones to manage cost exposure")
        if case_data.get('claim_amount', 0) > 1000000:
            explanation_parts.append("• Obtain independent case assessment before committing significant capital")
        if case_data.get('defendant_type') == 'Corporate':
            explanation_parts.append("• Verify defendant's financial position and insurance coverage")
        explanation_parts.append("• Ensure ATE insurance is in place to cover adverse costs risk")
        explanation_parts.append("• Require regular case progress reporting from instructed solicitors")

        return {
            'explanation': '\n'.join(explanation_parts),
            'source': 'rules',
            'model': 'rule-based',
            'timestamp': datetime.now().isoformat()
        }

    def _get_risk_detail(self, factor: str) -> str:
        """Get detailed explanation for a risk factor"""
        risk_details = {
            'high complexity': 'Complex cases require more resources, take longer, and have less predictable outcomes. Budget 20-30% contingency.',
            'high value claim': 'Claims over £1M typically face well-funded defence teams and may require specialist counsel. Expect increased costs.',
            'appellate': 'Appeal courts show greater deference to first instance findings. Success rates drop significantly at appellate level.',
            'lower success rate': 'Historical data suggests this case type has below-average success rates. Weight prediction accordingly.',
            'well-resourced defendant': 'Defendants with substantial resources can extend litigation and increase costs through procedural tactics.',
            'vigorous defense': 'High-value claims attract aggressive defence strategies, increasing both duration and cost.',
            'uncertainty': 'Multiple variables create wider confidence intervals on outcome prediction.'
        }

        factor_lower = factor.lower()
        for key, detail in risk_details.items():
            if key in factor_lower:
                return detail

        return "Consider this factor in your overall risk assessment."

    def detect_anomalies(self, cases: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """
        Detect anomalous cases that may need attention.
        Identifies outliers, inconsistencies, or unusual patterns.
        """
        anomalies = []

        if len(cases) < 3:
            return anomalies

        # Calculate averages
        amounts = [c.get('claim_amount', 0) for c in cases]
        avg_amount = sum(amounts) / len(amounts)

        success_rates = [p.get('success_rate', 50) for p in predictions]
        avg_success = sum(success_rates) / len(success_rates)

        for case, pred in zip(cases, predictions):
            case_anomalies = []

            # Check for amount outliers
            amount = case.get('claim_amount', 0)
            if amount > avg_amount * 5:
                case_anomalies.append({
                    'type': 'high_value_outlier',
                    'description': f'Claim amount (£{amount:,.0f}) is {amount/avg_amount:.1f}x higher than portfolio average',
                    'severity': 'warning'
                })

            # Check for success rate anomalies
            success = pred.get('success_rate', 50)
            if success > 85:
                case_anomalies.append({
                    'type': 'unusually_high_success',
                    'description': f'Predicted success rate ({success}%) is unusually high - verify assumptions',
                    'severity': 'info'
                })
            elif success < 30:
                case_anomalies.append({
                    'type': 'very_low_success',
                    'description': f'Very low predicted success ({success}%) - consider if funding is appropriate',
                    'severity': 'warning'
                })

            # Check for inconsistencies
            if case.get('complexity') == 'Low' and pred.get('risk_level') == 'High':
                case_anomalies.append({
                    'type': 'complexity_risk_mismatch',
                    'description': 'Low complexity but high risk - unusual combination, verify case details',
                    'severity': 'info'
                })

            if case.get('case_type') == 'Debt Recovery' and case.get('complexity') == 'High':
                case_anomalies.append({
                    'type': 'type_complexity_mismatch',
                    'description': 'Debt recovery cases are typically straightforward - high complexity is unusual',
                    'severity': 'info'
                })

            if case_anomalies:
                anomalies.append({
                    'case_id': case.get('case_id'),
                    'case_type': case.get('case_type'),
                    'anomalies': case_anomalies
                })

        return anomalies

    def generate_portfolio_insights(self, cases: List[Dict], predictions: List[Dict]) -> Dict:
        """
        Generate AI-powered insights for the entire portfolio.
        """
        if not cases:
            return {'insights': 'No cases in portfolio to analyze.', 'source': 'rules'}

        # Calculate metrics
        total_exposure = sum(c.get('claim_amount', 0) for c in cases)
        avg_success = sum(p.get('success_rate', 0) for p in predictions) / len(predictions) if predictions else 0

        # Count distributions
        type_dist = {}
        risk_dist = {'Low': 0, 'Medium': 0, 'High': 0}

        for case in cases:
            ct = case.get('case_type', 'Unknown')
            type_dist[ct] = type_dist.get(ct, 0) + 1

        for pred in predictions:
            rl = pred.get('risk_level', 'Medium')
            risk_dist[rl] = risk_dist.get(rl, 0) + 1

        if self.client:
            prompt = f"""Analyze this litigation funding portfolio and provide strategic insights:

Portfolio Summary:
- Total Cases: {len(cases)}
- Total Exposure: £{total_exposure:,.0f}
- Average Success Rate: {avg_success:.1f}%

Case Type Distribution:
{chr(10).join(f'- {k}: {v} cases' for k, v in type_dist.items())}

Risk Distribution:
- Low Risk: {risk_dist['Low']} cases
- Medium Risk: {risk_dist['Medium']} cases
- High Risk: {risk_dist['High']} cases

Provide:
1. Portfolio health assessment
2. Concentration risk analysis
3. Diversification recommendations
4. Key concerns or opportunities
5. Strategic recommendations

Be concise but insightful."""

            llm_response = self._call_llm(prompt)
            if llm_response:
                return {
                    'insights': llm_response,
                    'source': 'ai',
                    'model': self.model,
                    'metrics': {
                        'total_cases': len(cases),
                        'total_exposure': total_exposure,
                        'avg_success': avg_success,
                        'risk_distribution': risk_dist
                    }
                }

        # Rule-based insights
        return self._rule_based_portfolio_insights(cases, predictions, type_dist, risk_dist, total_exposure, avg_success)

    def _rule_based_portfolio_insights(self, cases, predictions, type_dist, risk_dist,
                                       total_exposure, avg_success) -> Dict:
        """Generate rule-based portfolio insights"""
        insights = []

        # Portfolio health
        if avg_success >= 65:
            insights.append(f"**Portfolio Health: GOOD** - Average predicted success rate of {avg_success:.1f}% indicates a healthy portfolio.")
        elif avg_success >= 50:
            insights.append(f"**Portfolio Health: MODERATE** - Average success rate of {avg_success:.1f}% suggests room for improvement in case selection.")
        else:
            insights.append(f"**Portfolio Health: CONCERNING** - Average success rate of {avg_success:.1f}% is below target. Review case selection criteria.")

        # Concentration risk
        if type_dist:
            max_type = max(type_dist.items(), key=lambda x: x[1])
            concentration = max_type[1] / len(cases) * 100
            if concentration > 50:
                insights.append(f"**Concentration Risk: HIGH** - {max_type[0]} represents {concentration:.0f}% of portfolio. Consider diversifying.")
            elif concentration > 30:
                insights.append(f"**Concentration Risk: MODERATE** - {max_type[0]} is the largest category at {concentration:.0f}%.")
            else:
                insights.append("**Concentration Risk: LOW** - Portfolio is well diversified across case types.")

        # Risk distribution
        high_risk_pct = risk_dist['High'] / len(cases) * 100 if cases else 0
        if high_risk_pct > 30:
            insights.append(f"**Risk Alert:** {high_risk_pct:.0f}% of cases are high risk. Consider rebalancing toward lower-risk opportunities.")
        elif high_risk_pct > 15:
            insights.append(f"**Risk Note:** {high_risk_pct:.0f}% high-risk cases - within acceptable range but monitor closely.")

        # Exposure
        avg_case_value = total_exposure / len(cases) if cases else 0
        insights.append(f"**Exposure Summary:** Total exposure £{total_exposure:,.0f} across {len(cases)} cases (avg £{avg_case_value:,.0f}/case).")

        # Recommendations
        insights.append("\n**Recommendations:**")
        if high_risk_pct > 20:
            insights.append("• Tighten case selection criteria to reduce high-risk exposure")
        if len(type_dist) < 3:
            insights.append("• Seek opportunities in under-represented case types for diversification")
        if avg_success < 60:
            insights.append("• Review cases with success rate below 50% for potential exit")
        insights.append("• Maintain regular portfolio reviews and stress testing")

        return {
            'insights': '\n'.join(insights),
            'source': 'rules',
            'model': 'rule-based',
            'metrics': {
                'total_cases': len(cases),
                'total_exposure': total_exposure,
                'avg_success': avg_success,
                'risk_distribution': risk_dist
            }
        }

    def get_status(self) -> Dict:
        """Get AI assistant status"""
        return {
            'provider': self.provider,
            'model': self.model,
            'llm_available': self.client is not None,
            'openai_installed': OPENAI_AVAILABLE,
            'anthropic_installed': ANTHROPIC_AVAILABLE
        }


# Singleton instance
_ai_assistant = None


def get_ai_assistant() -> AIAssistant:
    """Get singleton AI assistant instance"""
    global _ai_assistant
    if _ai_assistant is None:
        _ai_assistant = AIAssistant()
    return _ai_assistant
