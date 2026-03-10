def relationship_label(domain, alignment):

    if domain > 70 and alignment > 70:
        return "Strong Strategic Alignment"

    if domain > 70 and alignment < 50:
        return "Competitive or Adversarial Relationship"

    if domain < 50 and alignment > 70:
        return "Shared Values but Different Domains"

    return "Low Strategic Connection"