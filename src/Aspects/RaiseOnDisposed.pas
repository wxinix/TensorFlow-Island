namespace TensorFlow.Island.Aspects;

interface

uses
  RemObjects.Elements.Cirrus;

type
  [AttributeUsage(AttributeTargets.Class)]
  RaiseOnDisposedAttribute = public class(Attribute, IMethodImplementationDecorator)
  public
    method HandleImplementation(Services: IServices; aMethod: IMethodDefinition);
  end;

implementation

method RaiseOnDIsposedAttribute.HandleImplementation(Services: IServices; aMethod: IMethodDefinition);
begin
  if 
    String.Equals(aMethod.Name, '.', StringComparison.OrdinalIgnoreCase) or
    String.Equals(aMethod.Name, '~', StringComparison.OrdinalIgnoreCase) or
    (aMethod.Visibility <> Visibility.Public)
  then begin
    exit;
  end;

  aMethod.SetBody(Services,
    method begin
      // Call DisposableObject.CheckAndRaiseOnDisposed
      Aspects.OriginalBody;
    end);  
end;

end.